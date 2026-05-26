"""Module 1: Dual-Path Gated Text Conductor.

Replaces the original 2-layer MLP adapter with two parallel pathways:
  - Local path:  per-token MLP (preserves the original MLP's role)
  - Global path: multi-head self-attention (captures cross-token relations)

The two paths are fused by a per-token, per-dim gate g that is conditioned
on BOTH the token itself AND a global sequence summary. This lets relational
tokens (e.g. "to the left of") get more global context while content tokens
(e.g. "red apple") stay local.

Math (per layer):
    T_local  = W2 · GELU(W1 · T)
    T_global = MHA(LN(T))
    g        = sigmoid(W_g · [T, mean(T)])              # ∈ [0,1]^(L×d)
    T̃        = LN(T + g ⊙ T_local + (1-g) ⊙ T_global)
"""

import torch
import torch.nn as nn


class DualPathGatedTextConductor(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # ── Local MLP path (same role as the original GOAT adapter) ──
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden, dim),
        )

        # ── Global Self-Attention path (new) ──
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # ── Context-aware Gate (new) ──
        # Input: concat([token, global_avg_summary]) of width 2d.
        # The global summary lets the gate see "what kind of sentence this is",
        # so relational tokens can preferentially route through Self-Attn.
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        T: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            T: [B, L, d] text token embeddings from the text encoder.
            key_padding_mask: [B, L] bool tensor; True at padded positions
                (follows PyTorch MHA convention). Optional.

        Returns:
            T_tilde: [B, L, d] generation-oriented text condition.
        """
        B, L, d = T.shape

        # Local path
        T_local = self.mlp(T)                                    # [B, L, d]

        # Global path
        T_norm = self.attn_norm(T)
        T_global, _ = self.attn(
            T_norm, T_norm, T_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                                         # [B, L, d]

        # Global sequence summary (mask-aware)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)    # [B, L, 1]
            T_avg = (T * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1e-6)
        else:
            T_avg = T.mean(dim=1)                                # [B, d]
        T_avg_expand = T_avg.unsqueeze(1).expand(-1, L, -1)      # [B, L, d]

        # Gate: per-token, per-dim ∈ [0,1]
        T_aug = torch.cat([T, T_avg_expand], dim=-1)             # [B, L, 2d]
        g = self.gate(T_aug)                                      # [B, L, d]

        # Gated fusion + residual
        delta = g * T_local + (1.0 - g) * T_global
        T_tilde = self.out_norm(T + delta)
        return T_tilde
