import torch
import torch.nn as nn


class AdaptiveFourierBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        spatial_dims: int = 1,        # 1: input [B, L, d]   2: input [B, H, W, d]
        hidden: int = None,
        use_band_split: bool = False,
        num_bands: int = 3,
    ):
        super().__init__()
        assert spatial_dims in (1, 2)
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.use_band_split = use_band_split
        self.num_bands = num_bands

        hidden = hidden if hidden is not None else max(dim // 2, 16)

        out_dim = dim * num_bands if use_band_split else dim
        self.mask_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    # --------------------------------------------------------------------- #
    # Band masks: radial frequency bands [num_bands, H, W]                  #
    # --------------------------------------------------------------------- #
    def _make_band_masks(self, H: int, W: int, device, dtype) -> torch.Tensor:
        u = torch.fft.fftfreq(H, device=device).reshape(-1, 1).expand(H, W)
        v = torch.fft.fftfreq(W, device=device).reshape(1, -1).expand(H, W)
        r = torch.sqrt(u ** 2 + v ** 2)                          # [H, W]
        r_max = r.max() + 1e-6

        masks = []
        for i in range(self.num_bands):
            lo = i * r_max / self.num_bands
            hi = (i + 1) * r_max / self.num_bands
            band = ((r >= lo) & (r < hi)).to(dtype)
            masks.append(band)
        return torch.stack(masks, dim=0)                         # [num_bands, H, W]

    # --------------------------------------------------------------------- #
    # Forward                                                               #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims == 1:
            return self._forward_1d(x)
        return self._forward_2d(x)

    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d]
        Fv = torch.fft.fft(x, dim=1)                              # complex [B, L, d]
        amp_stat = Fv.abs().mean(dim=1)                           # [B, d]
        M_dyn = torch.sigmoid(self.mask_mlp(amp_stat))            # [B, d]
        M_dyn = M_dyn.unsqueeze(1)                                # [B, 1, d]
        Fv_filtered = M_dyn * Fv
        return torch.fft.ifft(Fv_filtered, dim=1).real

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, d]
        B, H, W, d = x.shape
        Fv = torch.fft.fft2(x, dim=(1, 2))                        # complex
        amp_stat = Fv.abs().mean(dim=(1, 2))                      # [B, d]

        if self.use_band_split:
            w = torch.sigmoid(self.mask_mlp(amp_stat))            # [B, d*N]
            w = w.view(B, self.num_bands, d)                      # [B, N, d]
            band_masks = self._make_band_masks(H, W, x.device, w.dtype)  # [N, H, W]

            # Combine: M_dyn[b, h, w, d] = Σ_n band_masks[n,h,w] * w[b,n,d]
            # einsum: 'nhw,bnd -> bhwd'
            M_dyn = torch.einsum('nhw,bnd->bhwd', band_masks, w)  # [B, H, W, d]
        else:
            M_dyn = torch.sigmoid(self.mask_mlp(amp_stat))        # [B, d]
            M_dyn = M_dyn.view(B, 1, 1, d).expand(B, H, W, d)

        Fv_filtered = M_dyn * Fv
        return torch.fft.ifft2(Fv_filtered, dim=(1, 2)).real
