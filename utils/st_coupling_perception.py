import torch
import torch.nn as nn


class SpatialTemporalCouplingPerception(nn.Module):
    def __init__(
        self,
        patch_len: int,
        time_steps: int = None,
        eps: float = 1e-6,
        add_self_loops: bool = True,
        return_layout: str = 'same',
    ):
        super().__init__()
        # Spatial-Temporal Coupling Perception (Section "Spatial-Temporal Coupling Perception")
        # This module implements:
        #  (1) Patch-wise Temporal Segmentation:  X -> {X^(p)}
        #  (2) Local Dynamic Graph Learning via PCC:  X^(p) -> A^(p)
        #  (3) Coupled Representation with GNN (GCN) + Aggregation:
        #      H^(p) = GNN(X^(p), A^(p)),  E_ST = Concat_p H^(p)
        if patch_len <= 0:
            raise ValueError('patch_len must be positive')
        if return_layout not in {'same', 'bln', 'bnl'}:
            raise ValueError("return_layout must be one of {'same','bln','bnl'}")

        self.patch_len = int(patch_len)
        self.time_steps = None if time_steps is None else int(time_steps)
        self.eps = float(eps)
        self.add_self_loops = bool(add_self_loops)
        self.return_layout = return_layout

    def _to_bnl(self, x: torch.Tensor):
        # Utility: make the internal representation consistent as [B, N, L]
        # where N is #variables and L is lookback length.
        if x.ndim != 3:
            raise ValueError('Expected a 3D tensor')
        b, d1, d2 = x.shape
        if self.time_steps is not None:
            if d1 == self.time_steps and d2 != self.time_steps:
                return x.permute(0, 2, 1).contiguous()
            if d2 == self.time_steps and d1 != self.time_steps:
                return x
        if d1 < d2:
            return x
        return x.permute(0, 2, 1).contiguous()

    @staticmethod
    def _to_bln(x_bnl: torch.Tensor):
        # Utility: convert [B, N, L] -> [B, L, N]
        return x_bnl.permute(0, 2, 1).contiguous()

    def _patchify(self, x_bnl: torch.Tensor):
        # Patch-wise Temporal Segmentation (Eq. (1)):
        # Given X in R^{N x L}, divide along time into P non-overlapping patches
        # X^(p) in R^{N x L'}, where P = floor(L / L').
        # Here we operate on batched input: x_bnl in R^{B x N x L}.
        b, n, l = x_bnl.shape
        p = l // self.patch_len
        if p <= 0:
            raise ValueError(f'Lookback length L={l} must be >= patch_len={self.patch_len}')
        if l % self.patch_len != 0:
            raise ValueError(f'Lookback length L={l} must be divisible by patch_len={self.patch_len}')
        l_eff = p * self.patch_len
        x_bnl = x_bnl[:, :, :l_eff]
        # Reshape to patched representation:
        # patches[b, :, p, :] corresponds to X^(p) for batch b.
        x = x_bnl.view(b, n, p, self.patch_len)
        return x, l_eff

    def _pcc_adjacency(self, x_bnl_patch: torch.Tensor):
        # Local Dynamic Graph Learning (Eq. (2)):
        # Compute patch-specific adjacency A^(p) using Pearson Correlation
        # between variable i and j restricted to the patch window.
        # Input:  x_bnl_patch in R^{B x N x L'} (a single patch per batch)
        # Output: corr in R^{B x N x N} (A^(p) for each batch)
        b, n, lp = x_bnl_patch.shape
        # Center each variable within the patch: x_{i,k} - mean_i
        xm = x_bnl_patch - x_bnl_patch.mean(dim=-1, keepdim=True)
        # Sample covariance: sum_k (...) / (L' - 1)
        denom = float(max(lp - 1, 1))
        cov = torch.matmul(xm, xm.transpose(-1, -2)) / denom
        # Sample variance for each variable i: sum_k (...)^2 / (L' - 1)
        var = (xm * xm).sum(dim=-1, keepdim=False) / denom
        std = torch.sqrt(var.clamp_min(self.eps))
        # Pearson correlation: cov(i,j) / (std(i) * std(j))
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + self.eps)
        corr = corr.clamp(-1.0, 1.0)
        return corr

    def _normalize_adj(self, a: torch.Tensor):
        # Implementation detail for the GNN in Eq. (3):
        # We use a simple GCN-style aggregation, so we compute a normalized
        # adjacency \hat{A} = D^{-1/2} (A + I) D^{-1/2}.
        b, n, _ = a.shape
        if self.add_self_loops:
            eye = torch.eye(n, device=a.device, dtype=a.dtype).unsqueeze(0)
            a = a + eye
        deg = a.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp_min(self.eps), -0.5)
        a_norm = a * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
        return a_norm

    def forward(self, x: torch.Tensor):
        # Full pipeline:
        #  - Eq. (1) patchify
        #  - Eq. (2) PCC adjacency per patch
        #  - Eq. (3) GNN/GCN fusion per patch
        #  - Eq. (4) Concat patches to restore length L
        x_in = x
        x_bnl = self._to_bnl(x)
        b, n, l = x_bnl.shape

        # Eq. (1): X -> {X^(p)}
        patches, l_eff = self._patchify(x_bnl)
        _, _, p, lp = patches.shape

        h_list = []
        for pi in range(p):
            # X^(p) in R^{B x N x L'}
            x_patch = patches[:, :, pi, :]
            # Eq. (2): A^(p) in R^{B x N x N}
            a = self._pcc_adjacency(x_patch)
            a_norm = self._normalize_adj(a)
            # Eq. (3): H^(p) = GNN(X^(p), A^(p))
            # Here GNN is a single GCN-like layer; treating the time steps (L')
            # as feature channels, we aggregate neighbors by left-multiplying A.
            h_patch = torch.matmul(a_norm, x_patch)
            h_list.append(h_patch)

        # Eq. (4): H_out = Concat(H^(1), ..., H^(P)) along time dimension
        h_bnl = torch.cat(h_list, dim=-1)
        if l_eff < l:
            # Note: if L is not divisible by L', we keep the tail segment unchanged
            # and append it back to preserve the original lookback length.
            pad = x_bnl[:, :, l_eff:]
            h_bnl = torch.cat([h_bnl, pad], dim=-1)

        # Output layout handling (keeps interface compatible with caller)
        if self.return_layout == 'bnl':
            return h_bnl
        if self.return_layout == 'bln':
            return self._to_bln(h_bnl)

        if x_in.shape == x_bnl.shape:
            return h_bnl
        return self._to_bln(h_bnl)
