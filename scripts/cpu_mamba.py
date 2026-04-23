"""CPU-only Mamba implementation whose state_dict is compatible with the
`NTU_Fi_Mamba` checkpoints saved under `model_pt/`. The upstream `mamba-ssm`
package requires CUDA and a working CUDA toolchain to build, which isn't
available on macOS; this module implements the selective SSM scan in plain
PyTorch so the demo bundle builder can run Mamba on CPU.

Correctness note: this is a straightforward Python-loop implementation of the
Mamba-1 selective scan (y[t] = C[t] · h[t] + D·x[t], with the discretized
state update h[t] = exp(dt[t]·A)·h[t-1] + dt[t]·B[t]·x[t]). It matches the
math exactly and is fine for small-batch / short-sequence inference; do not
use it for training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CpuSelectiveMamba(nn.Module):
    """Drop-in replacement for ``mamba_ssm.Mamba`` at inference time."""

    def __init__(self, d_model: int = 256, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, _ = x.shape
        xz = self.in_proj(x)                            # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                   # each (B, L, d_inner)

        # 1D causal conv along time
        x_in = x_in.transpose(1, 2)                     # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :L]               # causal via trailing truncation
        x_in = x_in.transpose(1, 2)                     # (B, L, d_inner)
        x_in = F.silu(x_in)

        # Selective projections
        x_db = self.x_proj(x_in)                        # (B, L, dt_rank + 2*d_state)
        dt, Bp, Cp = torch.split(
            x_db,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))               # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())              # (d_inner, d_state)

        # Selective scan, done in Python for CPU correctness.
        deltaA = torch.exp(dt.unsqueeze(-1) * A)        # (B, L, d_inner, d_state)
        deltaBx = dt.unsqueeze(-1) * Bp.unsqueeze(-2) * x_in.unsqueeze(-1)
        # deltaBx: (B, L, d_inner, d_state)
        h = x_in.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaBx[:, t]        # (B, d_inner, d_state)
            y = (h * Cp[:, t].unsqueeze(1)).sum(-1)     # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)                      # (B, L, d_inner)
        y = y + x_in * self.D

        y = y * F.silu(z)
        return self.out_proj(y)                         # (B, L, d_model)


class CpuMambaEncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = CpuSelectiveMamba(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.mamba(self.norm(x))
        return residual + self.dropout(x)


class CpuGenericMambaClassifier(nn.Module):
    """Matches the state_dict layout of `wisense.models.mamba_generic.GenericMambaClassifier`
    with selective=True, so it can load the saved Mamba checkpoints directly."""

    def __init__(
        self,
        num_classes: int,
        in_features: int,
        d_model: int = 256,
        depth: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            CpuMambaEncoderBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Caller passes (B, 3, 114, T). Flatten antennas × subcarriers, then
        # put time on the sequence dim to match upstream shape (B, T, D).
        if x.dim() == 4:
            b, c, s, t = x.shape
            x = x.reshape(b, c * s, t).transpose(1, 2)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, -1, :]
        return self.head(x)
