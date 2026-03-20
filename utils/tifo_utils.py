
import math
import os
from functools import partial
from turtle import forward
from typing import Optional, Tuple, Union

from IPython import embed
from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, drop_path
from torch.utils.checkpoint import checkpoint

# Import flash_attn's attention
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc

from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc

def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class CrossAttention(nn.Module):
    def __init__(
        self, for_tifo=True, num_slots=6, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False
    ):
        cos_attn=False
        super().__init__()
        self.for_tifo = for_tifo
        self.num_slots = num_slots
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.cos_attn = cos_attn
        
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)

        if for_tifo:
            q = torch.empty(num_slots, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)

        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
    
    def forward(self, q, ca_kv):
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]

        kv_compact = F.linear(
            kv_compact,
            weight=self.mat_kv.weight,
            bias=torch.cat((self.zero_k_bias, self.v_bias))
        ).view(N, 2, self.num_heads, self.head_dim)   # [N, 2, H, Dh]

        if not self.for_tifo:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(B * Lq, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = self.num_slots

            # self.mat_q: [K, H, Dh]
            # -> [B, K, H, Dh]
            # -> [B*K, H, Dh]
            q_compact = (
                self.mat_q.unsqueeze(0)
                .repeat(B, 1, 1, 1)
                .reshape(B * Lq, self.num_heads, self.head_dim)
                .to(dtype=kv_compact.dtype, device=kv_compact.device)
            )

        if self.cos_attn:
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)

        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()

        # 普通模式: 每个样本有 Lq 个 query
        # slot模式: 每个样本有 num_slots 个 query
        cu_seqlens_q = torch.arange(
            0, Lq * (B + 1), Lq,
            dtype=torch.int32,
            device=q_compact.device
        )

        if q_compact.dtype == torch.float32:
            oup = flash_attn_varlen_kvpacked_func(
                q=q_compact.to(dtype=torch.bfloat16),
                kv=kv_compact.to(dtype=torch.bfloat16),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=Lq,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0,
                softmax_scale=self.scale
            ).reshape(B, Lq, -1).float()
        else:
            oup = flash_attn_varlen_kvpacked_func(
                q=q_compact,
                kv=kv_compact,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=Lq,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0,
                softmax_scale=self.scale
            ).reshape(B, Lq, -1)

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, num_slots={self.num_slots}, cos_attn={self.cos_attn}'
    

class SlotsAdapter(nn.Module):
    def __init__(self, embed_dim: int, num_slots: int):
        super().__init__()
        self.D = embed_dim
        if embed_dim > 4096:
            self.head_dim = 64
        else:
            self.head_dim = 128
        
        self.num_heads = embed_dim // self.head_dim
        self.ca = CrossAttention(
            num_slots=num_slots,
            for_tifo=True,
            embed_dim=self.D,
            kv_dim=embed_dim,
            num_heads=self.num_heads
        )
    def forward(self, ca_kv):
        return self.ca(None, ca_kv)


# 让每个slot不要太像

def calculate_div_loss(output):
    slots = output
    slots_norm = F.normalize(slots, dim=-1)
    sim = torch.matmul(slots_norm, slots_norm.transpose(-1, -2))  # [B, K, K]
    div_loss = sim.mean()  # 或者只惩罚非对角

    return div_loss