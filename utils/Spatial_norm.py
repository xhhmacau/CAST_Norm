import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CommunityAwareSpatialPurification(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        time_steps: int,
        num_communities: int,
        attn_dim: int = 64,
        gcn_dim: int = 64,
        momentum: float = 0.99,
        eps: float = 1e-6,
        add_self_loops: bool = True,
    ):
        super().__init__()
        if num_nodes <= 0:
            raise ValueError('num_nodes must be positive')
        if time_steps <= 0:
            raise ValueError('time_steps must be positive')
        if num_communities <= 0:
            raise ValueError('num_communities must be positive')
        if not (0.0 <= momentum <= 1.0):
            raise ValueError('momentum must be in [0, 1]')

        self.num_nodes = int(num_nodes)
        self.time_steps = int(time_steps)
        self.num_communities = int(num_communities)
        self.attn_dim = int(attn_dim)
        self.gcn_dim = int(gcn_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.add_self_loops = bool(add_self_loops)

        # Invariant extractor f_inv(·): applied per node over the time dimension.
        # Current setting: x_inv = f_inv(x), without explicit residual x + f(x).
        # We initialize it as identity so that x_inv ≈ x at the beginning of training.
        self.inv_mlp = nn.Sequential(
            nn.Linear(self.time_steps, self.time_steps),
            nn.Identity(),
            nn.Linear(self.time_steps, self.time_steps),
        )
        nn.init.eye_(self.inv_mlp[0].weight)
        nn.init.zeros_(self.inv_mlp[0].bias)
        nn.init.eye_(self.inv_mlp[-1].weight)
        nn.init.zeros_(self.inv_mlp[-1].bias)

        # Steady-state anchor topology M_st in Eq. (1): shape [N, N].
        # It is updated by momentum and should NOT receive gradients.
        self.register_buffer('m_st', torch.zeros(self.num_nodes, self.num_nodes))
        self.register_buffer('_m_st_initialized', torch.tensor(False))

        # -------- Step 3: Soft community assignment (Eq. (6) in CASD Step 3) --------
        # H_t = ReLU( D^{-1/2} A D^{-1/2} X W_gcn )
        self.w_gcn = nn.Linear(self.time_steps, self.gcn_dim, bias=False)
        # S_t = Softmax( H_t W_pool ) in Eq. (6)
        self.w_pool = nn.Linear(self.gcn_dim, self.num_communities, bias=True)

    def _corrcoef(self, x_bnl: torch.Tensor) -> torch.Tensor:
        # Corr(X) used by:
        #  - Step 1 momentum update anchor:  M_st^{(t)} = alpha M_st^{(t-1)} + (1-alpha) A
        #  - Consistency loss: || Corr(X_inv) - M_st^{t-1} ||_F^2
        # Input: x_bnl in R^{B x N x L}
        # Output: corr in R^{B x N x N}
        b, n, l = x_bnl.shape
        xm = x_bnl - x_bnl.mean(dim=-1, keepdim=True)
        denom = float(max(l - 1, 1))
        cov = torch.matmul(xm, xm.transpose(-1, -2)) / denom
        var = (xm * xm).sum(dim=-1) / denom
        std = torch.sqrt(var.clamp_min(self.eps))
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + self.eps)
        return corr.clamp(-1.0, 1.0)

    def _corrcoef_window(self, x_bnl: torch.Tensor) -> torch.Tensor:
        # Corr(X) where X is treated as a single window matrix in R^{N x L}.
        # In training we receive a batch of windows; to obtain a single A_t in R^{N x N}
        # (as in the paper), we concatenate samples along the time axis.
        # Input: x_bnl in R^{B x N x L}
        # Output: corr in R^{N x N}
        if x_bnl.ndim != 3:
            raise ValueError('Expected input with shape [B, N, L]')
        b, n, l = x_bnl.shape
        x_n_t = x_bnl.transpose(0, 1).contiguous().view(n, b * l)  # [N, B*L]
        xm = x_n_t - x_n_t.mean(dim=-1, keepdim=True)
        denom = float(max(b * l - 1, 1))
        cov = torch.matmul(xm, xm.transpose(-1, -2)) / denom
        var = (xm * xm).sum(dim=-1) / denom
        std = torch.sqrt(var.clamp_min(self.eps))
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + self.eps)
        return corr.clamp(-1.0, 1.0)

    def _normalize_adj(self, a: torch.Tensor) -> torch.Tensor:
        # Symmetric normalization for GCN (used in Step 3 Eq. (6)):
        # \hat{A} = D^{-1/2} (A + I) D^{-1/2}
        b, n, _ = a.shape
        if self.add_self_loops:
            eye = torch.eye(n, device=a.device, dtype=a.dtype).unsqueeze(0)
            a = a + eye
        deg = a.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp_min(self.eps), -0.5)
        a_norm = a * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
        return a_norm

    def _update_momentum_anchor(self, a_curr: torch.Tensor) -> torch.Tensor:
        # Eq. (1): M_st^{(t)} = alpha M_st^{(t-1)} + (1-alpha) A^{(p)}
        # In code, A^{(p)} is computed from the current window (Corr(X)).
        # We use batch-average adjacency as the update signal.
        if a_curr.ndim == 3:
            a_mean = a_curr.mean(dim=0).detach()
        elif a_curr.ndim == 2:
            a_mean = a_curr.detach()
        else:
            raise ValueError('Expected a_curr with shape [B, N, N] or [N, N]')

        if not bool(self._m_st_initialized.item()):
            self.m_st.copy_(a_mean)
            self._m_st_initialized.fill_(True)
            return self.m_st

        self.m_st.mul_(self.momentum).add_(a_mean * (1.0 - self.momentum))
        return self.m_st

    def _consistency_loss(self, x_inv: torch.Tensor, m_st_prev: torch.Tensor) -> torch.Tensor:
        # Eq. (3): L_cons = || Corr(X_inv) - M_st^{t-1} ||_F^2
        corr_inv = self._corrcoef(x_inv)
        diff = corr_inv - m_st_prev.unsqueeze(0)
        return (diff * diff).mean()


    def forward(self, x_bnl: torch.Tensor):
        # Input:
        #  x_bnl: [B, N, L]  (batch, node, time)
        # Output:
        #  x_inv: [B, N, L] invariant pattern
        #  x_va:  [B, N, L] variant pattern
        #  orth_loss, mincut_loss, consistency_loss
        if x_bnl.ndim != 3:
            raise ValueError('Expected input with shape [B, N, L]')
        b, n, l = x_bnl.shape
        if n != self.num_nodes:
            raise ValueError(f'num_nodes mismatch: got N={n}, expected {self.num_nodes}')
        if l != self.time_steps:
            raise ValueError(f'time_steps mismatch: got L={l}, expected {self.time_steps}')

        # ---- Topology memory (EMA) ----
        # A_t = Corr(X)  (per-sample adjacency for the current window)
        # M_st^t = alpha * M_st^{t-1} + (1-alpha) * A_t
        a_curr = self._corrcoef(x_bnl)  # [B, N, N]

        # Cache M_st^{t-1} for loss computation (stop-gradient)
        m_st_prev = self.m_st.detach().clone()

        # ---- Step 1: Invariant extraction (current ablation: no attention) ----
        # x_inv = f_inv(x)
        # x_inv, x are both in R^{B x N x L}
        x_inv = self.inv_mlp(x_bnl)  # [B, N, L]

        # ---- Step 2: Differential Signal (Eq. (4)) ----
        # x_diff = x - x_inv
        x_diff = x_bnl - x_inv  # [B, N, L]

        # ---- Step 3: Variant Pattern Purification (Eq. (6)-(10)) ----
        # Eq. (6): H_t = ReLU( \hat{A} X W_gcn )
        # a_norm = self._normalize_adj(a_curr)
        # Current setting: community assignment is guided by stable memory M_st^{t-1}.
        a_norm = self._normalize_adj(m_st_prev.unsqueeze(0).expand(b, -1, -1))
        h = torch.matmul(a_norm, x_bnl)  # [B, N, L]
        h = F.relu(self.w_gcn(h))  # [B, N, gcn_dim]

        # Eq. (6): S_t = Softmax( H_t W_pool )  (row-stochastic over K)
        s_logits = self.w_pool(h)  # [B, N, K]
        s = torch.softmax(s_logits, dim=-1)  # [B, N, K]

        # Eq. (7) + Eq. (8): mincut + orth losses on S, guided by M_st^{t-1}
        loss_mincut = self._mincut_loss(s, m_st_prev)
        loss_orth = self._orth_loss(s)

        # Eq. (9): C_va = Normalize( S^T X_diff )
        # Current setting: we do NOT apply L2-normalization on C_va.
        c_va = torch.matmul(s.transpose(-1, -2), x_diff)  # [B, K, L]
        c_va = F.normalize(c_va, p=2, dim=-1, eps=self.eps)  # [B, K, L]

        # Eq. (10): X_va = S C_va
        x_va = torch.matmul(s, c_va)  # [B, N, L]

        # Eq. (3): consistency constraint computed on the extracted invariant pattern.
        # L_cons = || Corr(x_inv) - StopGrad(M_st^{t-1}) ||_F^2
        loss_cons = self._consistency_loss(x_inv, m_st_prev)

        # ---- Update momentum anchor (Eq. (1)) ----
        # Important: update AFTER computing losses that use M_st^{t-1}.
        if self.training:
            self._update_momentum_anchor(a_curr)

        return x_inv, x_va, loss_orth, loss_mincut, loss_cons
