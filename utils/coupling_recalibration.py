import torch
import torch.nn as nn
import torch.nn.functional as F


if False:
    class CouplingAwareRecalibration(nn.Module):
        def __init__(
            self,
            pred_len: int,
            d_k: int = 32,
            hidden_dim: int = 64,
            eps: float = 1e-6,
        ):
            super().__init__()
            if pred_len <= 0:
                raise ValueError('pred_len must be positive')
            if d_k <= 0:
                raise ValueError('d_k must be positive')
            if hidden_dim <= 0:
                raise ValueError('hidden_dim must be positive')

            self.pred_len = int(pred_len)
            self.d_k = int(d_k)
            self.hidden_dim = int(hidden_dim)
            self.eps = float(eps)

            self.mlp_time = nn.Sequential(
                nn.Linear(3, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.pred_len * self.d_k),
            )
            self.mlp_space = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.pred_len * self.d_k),
            )
            self.mlp_gate = nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.d_k),
            )

            self.proj_mu = nn.Linear(self.d_k, 1)
            self.proj_sigma = nn.Linear(self.d_k, 1)

            nn.init.zeros_(self.proj_mu.weight)
            nn.init.zeros_(self.proj_mu.bias)
            nn.init.zeros_(self.proj_sigma.weight)
            nn.init.zeros_(self.proj_sigma.bias)

        def forward(
            self,
            e_st_bnl: torch.Tensor,
            x_va_bnl: torch.Tensor,
            mu: torch.Tensor,
            theta: torch.Tensor,
        ):
            if e_st_bnl.ndim != 3 or x_va_bnl.ndim != 3:
                raise ValueError('Expected e_st_bnl and x_va_bnl with shape [B, N, L]')
            if e_st_bnl.shape != x_va_bnl.shape:
                raise ValueError('e_st_bnl and x_va_bnl must have the same shape [B, N, L]')

            if mu.ndim == 2:
                mu = mu.unsqueeze(-1)
            if theta.ndim == 2:
                theta = theta.unsqueeze(-1)
            if mu.ndim != 3 or theta.ndim != 3:
                raise ValueError('Expected mu and theta with shape [B, N, 1] (or [B, N])')
            if mu.shape[:2] != e_st_bnl.shape[:2] or theta.shape[:2] != e_st_bnl.shape[:2]:
                raise ValueError('mu/theta batch and node dims must match e_st_bnl')
            if mu.shape[-1] != 1 or theta.shape[-1] != 1:
                raise ValueError('mu and theta must have last dim = 1')

            e_pool = e_st_bnl.mean(dim=-1, keepdim=False)
            xva_pool = x_va_bnl.mean(dim=-1, keepdim=False)

            mu_in = mu.squeeze(-1)
            sigma_in = theta.squeeze(-1).clamp_min(self.eps)

            i_time = torch.stack([mu_in, sigma_in, e_pool], dim=-1)
            f_time = self.mlp_time(i_time).view(i_time.shape[0], i_time.shape[1], self.pred_len, self.d_k)

            i_space = torch.stack([xva_pool, e_pool], dim=-1)
            f_space = self.mlp_space(i_space).view(i_space.shape[0], i_space.shape[1], self.pred_len, self.d_k)

            gate_in = e_pool.unsqueeze(-1)
            lam = torch.sigmoid(self.mlp_gate(gate_in)).unsqueeze(-2)

            f_dist = lam * f_time + (1.0 - lam) * f_space

            delta_mu = self.proj_mu(f_dist).squeeze(-1)
            log_sigma_scale = self.proj_sigma(f_dist).squeeze(-1).clamp(-5.0, 5.0)
            sigma_scale = torch.exp(log_sigma_scale)

            mu_final = mu_in.unsqueeze(-1).expand(-1, -1, self.pred_len) + delta_mu
            sigma_final = sigma_in.unsqueeze(-1).expand(-1, -1, self.pred_len) * sigma_scale

            return mu_final, sigma_final


class CouplingAwareRecalibration_v2(nn.Module):
    def __init__(
        self,
        pred_len: int,
        d_k: int = 32,
        hidden_dim: int = 64,
        eps: float = 1e-6,
        max_log_scale: float = 1.5,
    ):
        super().__init__()
        if pred_len <= 0:
            raise ValueError('pred_len must be positive')
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')

        self.pred_len = int(pred_len)
        self.d_k = int(d_k)
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.max_log_scale = float(max_log_scale)

        self.in_dim = 6

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2 * self.pred_len),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.delta_mu_gain = nn.Parameter(torch.tensor(0.1))
        self.log_scale_gain = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        e_st_bnl: torch.Tensor,
        x_va_bnl: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
    ):
        if e_st_bnl.ndim != 3 or x_va_bnl.ndim != 3:
            raise ValueError('Expected e_st_bnl and x_va_bnl with shape [B, N, L]')
        if e_st_bnl.shape != x_va_bnl.shape:
            raise ValueError('e_st_bnl and x_va_bnl must have the same shape [B, N, L]')

        if mu.ndim == 2:
            mu = mu.unsqueeze(-1)
        if theta.ndim == 2:
            theta = theta.unsqueeze(-1)
        if mu.ndim != 3 or theta.ndim != 3:
            raise ValueError('Expected mu and theta with shape [B, N, 1] (or [B, N])')
        if mu.shape[:2] != e_st_bnl.shape[:2] or theta.shape[:2] != e_st_bnl.shape[:2]:
            raise ValueError('mu/theta batch and node dims must match e_st_bnl')
        if mu.shape[-1] != 1 or theta.shape[-1] != 1:
            raise ValueError('mu and theta must have last dim = 1')

        mu_in = mu.squeeze(-1)
        theta_in = theta.squeeze(-1).clamp_min(self.eps)

        xva_mean = x_va_bnl.mean(dim=-1)
        xva_std = x_va_bnl.std(dim=-1, unbiased=False).clamp_min(self.eps)
        est_mean = e_st_bnl.mean(dim=-1)
        est_std = e_st_bnl.std(dim=-1, unbiased=False).clamp_min(self.eps)

        log_theta = torch.log(theta_in)

        feat = torch.stack([xva_mean, xva_std, est_mean, est_std, mu_in, log_theta], dim=-1)
        out = self.mlp(feat)
        delta_raw, log_scale_raw = torch.split(out, self.pred_len, dim=-1)

        delta_mu = self.delta_mu_gain * theta_in.unsqueeze(-1) * torch.tanh(delta_raw)
        log_scale = self.log_scale_gain * torch.tanh(log_scale_raw)
        log_scale = log_scale.clamp(-self.max_log_scale, self.max_log_scale)
        sigma_scale = torch.exp(log_scale)

        mu_final = mu_in.unsqueeze(-1) + delta_mu
        sigma_final = theta_in.unsqueeze(-1) * sigma_scale

        return mu_final, sigma_final


if False:
    class CouplingAwareRecalibration(nn.Module):
        def __init__(
            self,
            pred_len: int,
            d_k: int = 32,
            hidden_dim: int = 64,
            eps: float = 1e-6,
            max_log_scale: float = 1.5,
        ):
            super().__init__()
            if pred_len <= 0:
                raise ValueError('pred_len must be positive')
            if hidden_dim <= 0:
                raise ValueError('hidden_dim must be positive')

            self.pred_len = int(pred_len)
            self.d_k = int(d_k)
            self.hidden_dim = int(hidden_dim)
            self.eps = float(eps)
            self.max_log_scale = float(max_log_scale)

            self.in_dim = 6

            self.mlp = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 2 * self.pred_len),
            )

            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

            self.delta_mu_gain = nn.Parameter(torch.tensor(0.1))
            self.log_scale_gain = nn.Parameter(torch.tensor(0.1))

        def forward(
            self,
            e_st_bnl: torch.Tensor,
            x_va_bnl: torch.Tensor,
            mu: torch.Tensor,
            theta: torch.Tensor,
        ):
            if e_st_bnl.ndim != 3 or x_va_bnl.ndim != 3:
                raise ValueError('Expected e_st_bnl and x_va_bnl with shape [B, N, L]')
            if e_st_bnl.shape != x_va_bnl.shape:
                raise ValueError('e_st_bnl and x_va_bnl must have the same shape [B, N, L]')

            if mu.ndim == 2:
                mu = mu.unsqueeze(-1)
            if theta.ndim == 2:
                theta = theta.unsqueeze(-1)
            if mu.ndim != 3 or theta.ndim != 3:
                raise ValueError('Expected mu and theta with shape [B, N, 1] (or [B, N])')
            if mu.shape[:2] != e_st_bnl.shape[:2] or theta.shape[:2] != e_st_bnl.shape[:2]:
                raise ValueError('mu/theta batch and node dims must match e_st_bnl')
            if mu.shape[-1] != 1 or theta.shape[-1] != 1:
                raise ValueError('mu and theta must have last dim = 1')

            mu_in = mu.squeeze(-1)
            theta_in = theta.squeeze(-1).clamp_min(self.eps)

            xva_mean = x_va_bnl.mean(dim=-1)
            xva_std = x_va_bnl.std(dim=-1, unbiased=False).clamp_min(self.eps)
            est_mean = e_st_bnl.mean(dim=-1)
            est_std = e_st_bnl.std(dim=-1, unbiased=False).clamp_min(self.eps)

            log_theta = torch.log(theta_in)

            feat = torch.stack([xva_mean, xva_std, est_mean, est_std, mu_in, log_theta], dim=-1)
            out = self.mlp(feat)
            delta_raw, log_scale_raw = torch.split(out, self.pred_len, dim=-1)

            delta_mu = self.delta_mu_gain * theta_in.unsqueeze(-1) * torch.tanh(delta_raw)
            log_scale = self.log_scale_gain * torch.tanh(log_scale_raw)
            log_scale = log_scale.clamp(-self.max_log_scale, self.max_log_scale)
            sigma_scale = torch.exp(log_scale)

            mu_final = mu_in.unsqueeze(-1) + delta_mu
            sigma_final = theta_in.unsqueeze(-1) * sigma_scale

            return mu_final, sigma_final


class CouplingAwareRecalibration(nn.Module):
    def __init__(
        self,
        pred_len: int,
        d_k: int = 32,
        hidden_dim: int = 64,
        eps: float = 1e-6,
        max_log_scale: float = 1.5,
        delta_mu_gain_init: float = 0.1,
        log_scale_gain_init: float = 0.1,
    ):
        super().__init__()
        if pred_len <= 0:
            raise ValueError('pred_len must be positive')
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')

        self.pred_len = int(pred_len)
        self.d_k = int(d_k)
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.max_log_scale = float(max_log_scale)

        self.mlp_time = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2 * self.pred_len),
        )
        self.mlp_space = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2 * self.pred_len),
        )
        self.mlp_gate = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.pred_len),
        )

        nn.init.zeros_(self.mlp_time[-1].weight)
        nn.init.zeros_(self.mlp_time[-1].bias)
        nn.init.zeros_(self.mlp_space[-1].weight)
        nn.init.zeros_(self.mlp_space[-1].bias)
        nn.init.zeros_(self.mlp_gate[-1].weight)
        nn.init.zeros_(self.mlp_gate[-1].bias)

        self.delta_mu_gain = nn.Parameter(torch.tensor(float(delta_mu_gain_init)))
        self.log_scale_gain = nn.Parameter(torch.tensor(float(log_scale_gain_init)))

    def forward(
        self,
        e_st_bnl: torch.Tensor,
        x_va_bnl: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
    ):
        if e_st_bnl.ndim != 3 or x_va_bnl.ndim != 3:
            raise ValueError('Expected e_st_bnl and x_va_bnl with shape [B, N, L]')
        if e_st_bnl.shape != x_va_bnl.shape:
            raise ValueError('e_st_bnl and x_va_bnl must have the same shape [B, N, L]')

        if mu.ndim == 2:
            mu = mu.unsqueeze(-1)
        if theta.ndim == 2:
            theta = theta.unsqueeze(-1)
        if mu.ndim != 3 or theta.ndim != 3:
            raise ValueError('Expected mu and theta with shape [B, N, 1] (or [B, N])')
        if mu.shape[:2] != e_st_bnl.shape[:2] or theta.shape[:2] != e_st_bnl.shape[:2]:
            raise ValueError('mu/theta batch and node dims must match e_st_bnl')
        if mu.shape[-1] != 1 or theta.shape[-1] != 1:
            raise ValueError('mu and theta must have last dim = 1')

        mu_in = mu.squeeze(-1)
        theta_in = theta.squeeze(-1).clamp_min(self.eps)

        e_mean = e_st_bnl.mean(dim=-1)
        e_std = e_st_bnl.std(dim=-1, unbiased=False).clamp_min(self.eps)

        xva_mean = x_va_bnl.mean(dim=-1)
        xva_std = x_va_bnl.std(dim=-1, unbiased=False).clamp_min(self.eps)

        log_theta = torch.log(theta_in)

        feat_time = torch.stack([mu_in, log_theta, e_mean, e_std], dim=-1)
        out_time = self.mlp_time(feat_time)
        delta_mu_raw_t, r_raw_t = torch.split(out_time, self.pred_len, dim=-1)

        feat_space = torch.stack([xva_mean, xva_std, e_mean, e_std], dim=-1)
        out_space = self.mlp_space(feat_space)
        delta_mu_raw_s, r_raw_s = torch.split(out_space, self.pred_len, dim=-1)

        gate_feat = torch.stack([e_mean, e_std], dim=-1)
        alpha = torch.sigmoid(self.mlp_gate(gate_feat))

        delta_mu_raw = alpha * delta_mu_raw_t + (1.0 - alpha) * delta_mu_raw_s
        r_raw = alpha * r_raw_t + (1.0 - alpha) * r_raw_s

        delta_mu = self.delta_mu_gain * theta_in.unsqueeze(-1) * torch.tanh(delta_mu_raw)
        # r = self.log_scale_gain * torch.tanh(r_raw)
        # r = r.clamp(-self.max_log_scale, self.max_log_scale)

        r_raw_tanh = torch.tanh(r_raw)
        r = self.log_scale_gain * r_raw_tanh * (self.max_log_scale / 1.5)
        r = r.clamp(-self.max_log_scale, self.max_log_scale)

        sigma_scale = torch.exp(r)

        mu_final = mu_in.unsqueeze(-1) + delta_mu
        sigma_final = theta_in.unsqueeze(-1) * sigma_scale

        return mu_final, sigma_final
