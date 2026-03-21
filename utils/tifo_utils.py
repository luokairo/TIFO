
import math
import os
from functools import partial
from turtle import forward
from typing import Optional, Tuple, Union

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


def pack_kv(x):
    """
    x: [B, L, C]
    return:
        kv_compact: [B*L, C]
        cu_seqlens_k: [B+1]
        max_seqlen_k: L
    """
    B, L, C = x.shape
    kv_compact = x.reshape(B * L, C).contiguous()
    cu_seqlens_k = torch.arange(
        0, (B + 1) * L, L,
        dtype=torch.int32,
        device=x.device
    )
    max_seqlen_k = L
    return kv_compact, cu_seqlens_k, max_seqlen_k

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

        kv_compact = kv_compact / (kv_compact.norm(dim=-1, keepdim=True).clamp(min=1.0))

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
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        print("kv_compact finite:", kv_compact.isfinite().all().item())
        print("kv_compact stats:", kv_compact.abs().max().item(), kv_compact.abs().mean().item())
        result = self.ca(None, ca_kv)
        print("ca output finite:", result.isfinite().all().item())
        return result


# 让每个slot不要太像

def calculate_div_loss(slots):
    # slots: [B, K, C]
    slots_norm = F.normalize(slots, dim=-1)

    sim = torch.matmul(slots_norm, slots_norm.transpose(-1, -2))  # [B, K, K]

    B, K, _ = sim.shape
    eye = torch.eye(K, device=sim.device).unsqueeze(0)  # [1, K, K]

    # 只惩罚非对角
    off_diag = sim * (1 - eye)

    div_loss = (off_diag ** 2).mean()
    return div_loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StableSlotsAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_slots: int = 6,
        num_heads: int = 32,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = False,
        dropout: float = 0.0,
        slot_init_scale: float = 0.02,
        use_ffn: bool = True,
        attn_fp32: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attn_fp32 = attn_fp32
        self.use_ffn = use_ffn
        self.debug = debug

        # learnable slots, 小初始化
        self.slots = nn.Parameter(torch.randn(1, num_slots, embed_dim) * slot_init_scale)

        # pre-norm
        self.norm_x = nn.LayerNorm(embed_dim)
        self.norm_slots = nn.LayerNorm(embed_dim)

        # q, k, v projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        # output proj
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_drop = nn.Dropout(dropout)

        # optional FFN block
        if use_ffn:
            hidden_dim = int(embed_dim * mlp_ratio)
            self.ffn_norm = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout),
            )

    def _stats(self, name, x):
        if not self.debug:
            return
        xf = x.float()
        print(
            f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} "
            f"finite={torch.isfinite(x).all().item()} "
            f"nan={torch.isnan(x).any().item()} "
            f"inf={torch.isinf(x).any().item()} "
            f"min={xf.min().item():.6e} "
            f"max={xf.max().item():.6e} "
            f"mean={xf.mean().item():.6e} "
            f"std={xf.std().item():.6e}"
        )

    def forward(self, x, attention_mask=None):
        """
        x: [B, L, C]
        attention_mask: [B, L], 1 for valid, 0 for padding
        """
        B, L, C = x.shape
        assert C == self.embed_dim

        dtype_in = x.dtype
        device = x.device

        self._stats("input_x", x)

        # pre-norm
        x_norm = self.norm_x(x)
        slots = self.slots.expand(B, -1, -1)
        slots_norm = self.norm_slots(slots)

        self._stats("x_norm", x_norm)
        self._stats("slots_param", self.slots)
        self._stats("slots_norm", slots_norm)

        # qkv projection
        q = self.q_proj(slots_norm)   # [B, K, C]
        k = self.k_proj(x_norm)       # [B, L, C]
        v = self.v_proj(x_norm)       # [B, L, C]

        self._stats("q_proj", q)
        self._stats("k_proj", k)
        self._stats("v_proj", v)

        # reshape to attention format
        q = q.view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, Dh]
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)               # [B, H, L, Dh]
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)               # [B, H, L, Dh]

        self._stats("q_before_attn", q)
        self._stats("k_before_attn", k)
        self._stats("v_before_attn", v)

        # attention mask -> key padding mask
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, L] -> [B, 1, 1, L]
            attn_mask = attention_mask[:, None, None, :].to(torch.bool)

        # 用 fp32 做 attention，稳定很多
        if self.attn_fp32:
            q_attn = q.float()
            k_attn = k.float()
            v_attn = v.float()
        else:
            q_attn = q
            k_attn = k
            v_attn = v

        if attn_mask is not None:
            # scaled_dot_product_attention 里 True 表示参与，False 表示mask掉
            out = F.scaled_dot_product_attention(
                q_attn,
                k_attn,
                v_attn,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q_attn,
                k_attn,
                v_attn,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

        self._stats("attn_out", out)

        # [B, H, K, Dh] -> [B, K, C]
        out = out.transpose(1, 2).contiguous().view(B, self.num_slots, C)
        out = out.to(dtype_in)

        # residual 1
        slots = slots + self.out_drop(self.out_proj(out))
        self._stats("after_residual1", slots)

        # residual 2: FFN
        if self.use_ffn:
            slots = slots + self.ffn(self.ffn_norm(slots))
            self._stats("after_ffn", slots)

        return slots