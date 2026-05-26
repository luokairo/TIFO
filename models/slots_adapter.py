"""Module 3: Iterative Gated Slot Adapter.

Replaces a single-pass Q-Former cross-attention with the original
Slot Attention mechanism (Locatello et al. 2020), enhanced with:
  - Slot-competitive attention: softmax over the SLOT dim (not feature dim)
  - T iterations of refinement
  - GRU-style gated state update per iteration
  - Optional inter-slot self-attention (slot↔slot coordination)

==========================================================================
The single most important change vs the original GOAT Q-Former:
==========================================================================
    Original Q-Former:  A = softmax(QK^T / sqrt(d), dim=-1)   # over L
    Ours (Slot-Attn):   A = softmax(QK^T / sqrt(d), dim=1)    # over K

    With softmax over K, Σ_k A[k,l] = 1, so every visual feature must
    be "claimed" by some slot → slots compete and specialize.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotGRUCell(nn.Module):
    """Standard GRU cell, applied independently per slot (no shared state)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin_r = nn.Linear(2 * dim, dim)
        self.lin_z = nn.Linear(2 * dim, dim)
        self.lin_h = nn.Linear(2 * dim, dim)

    def forward(self, Q: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        # Q, updates: [B, K, d]
        cat = torch.cat([Q, updates], dim=-1)
        r = torch.sigmoid(self.lin_r(cat))
        z = torch.sigmoid(self.lin_z(cat))

        h_in = torch.cat([r * Q, updates], dim=-1)
        h_tilde = torch.tanh(self.lin_h(h_in))

        Q_new = (1.0 - z) * Q + z * h_tilde
        return Q_new


class IterativeGatedSlotAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_iters: int = 2,
        use_slot_self_attn: bool = True,
        attn_heads: int = 4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.eps = eps

        # Learnable initial slot queries; small init to keep training stable
        self.slots_init = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)

        # Pre-norms (input features and slot states)
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

        # Q/K/V projections for slot-competitive attention
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

        # GRU update
        self.gru = SlotGRUCell(dim)

        # Optional slot↔slot self-attention
        self.use_slot_self_attn = use_slot_self_attn
        if use_slot_self_attn:
            self.slot_attn_norm = nn.LayerNorm(dim)
            self.slot_self_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=attn_heads, batch_first=True,
            )

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            V: [B, L, d] input features (visual or text).
        Returns:
            S: [B, K, d] slot representations.
        """
        B = V.size(0)
        V_norm = self.norm_inputs(V)
        k = self.to_k(V_norm)                                    # [B, L, d]
        v = self.to_v(V_norm)                                    # [B, L, d]

        # Initialize slots (broadcast across batch)
        Q = self.slots_init.expand(B, -1, -1).contiguous()       # [B, K, d]

        for _ in range(self.num_iters):
            q = self.to_q(self.norm_slots(Q))                    # [B, K, d]

            # Attention logits: [B, K, L]
            attn_logits = torch.einsum('bkd,bld->bkl', q, k) * self.scale

            # === Slot competition: softmax over slot dim K ===
            attn = F.softmax(attn_logits, dim=1)                 # [B, K, L]

            # Per-slot renormalization (Slot Attention trick): makes the
            # update a weighted MEAN rather than a weighted sum, so the
            # update scale doesn't blow up with sequence length.
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            updates = torch.einsum('bkl,bld->bkd', attn_norm, v)  # [B, K, d]

            # GRU-gated state update
            Q = self.gru(Q, updates)

            # Optional: slots see each other to reduce redundancy
            if self.use_slot_self_attn:
                Q_pre = self.slot_attn_norm(Q)
                Q_msg, _ = self.slot_self_attn(
                    Q_pre, Q_pre, Q_pre, need_weights=False,
                )
                Q = Q + Q_msg

        return self.norm_out(Q)
