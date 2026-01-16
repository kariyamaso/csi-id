import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba as MambaLayer
except ImportError:  # pragma: no cover - optional dependency
    MambaLayer = None


class DiagonalStateSpaceLayer(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1, use_skip: bool = True):
        super().__init__()
        self.d_state = d_state
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.use_skip = bool(use_skip)
        self.skip = nn.Linear(d_model, d_model, bias=False) if self.use_skip else None
        self.log_decay = nn.Parameter(torch.zeros(d_state))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        state = x.new_zeros(batch_size, self.d_state)
        decay = torch.sigmoid(self.log_decay).unsqueeze(0)
        outputs = []
        for t in range(seq_len):
            ut = x[:, t, :]
            state = state * decay + self.B(ut)
            yt = self.C(self.activation(state))
            outputs.append(yt.unsqueeze(1))
        y = torch.cat(outputs, dim=1)
        if self.use_skip:
            y = y + self.skip(x)
        return self.dropout(y)


class MambaEncoderBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        if MambaLayer is None:
            raise ImportError("mamba-ssm is required for Mamba models. Install it via `pip install mamba-ssm`.")
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.mamba(self.norm(x))
        return residual + self.dropout(x)


class NonSelectiveMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagonalStateSpaceLayer(d_model, d_state, dropout, use_skip=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ssm(self.norm(x))
        return residual + self.dropout(x)


class GenericMambaClassifier(nn.Module):
    """Mamba classifier for generic tensors.

    Supports:
    - CSI-like: (B, 3, 114, T) -> (B, T, 342)
    - UT-HAR:   (B, 1, 250, 90) -> (B, 250, 90)
    - Widar:    (B, 22, 20, 20) -> (B, 20, 440)
    """

    def __init__(
        self,
        num_classes: int,
        in_features: int,
        *,
        d_model: int = 256,
        depth: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
        selective: bool = True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.pooling = pooling
        self.pool_attn = nn.Linear(d_model, 1) if pooling == "attn" else None
        self.input_proj = nn.Sequential(nn.Linear(self.in_features, d_model), nn.LayerNorm(d_model))
        block_cls = MambaEncoderBlock if selective else NonSelectiveMambaBlock
        self.blocks = nn.ModuleList(
            [
                (
                    block_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
                    if selective
                    else block_cls(d_model=d_model, d_state=d_state, dropout=dropout)
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def _pool(self, seq):
        if self.pooling == "mean":
            return seq.mean(dim=1)
        if self.pooling == "last":
            return seq[:, -1, :]
        if self.pooling == "attn":
            weights = torch.softmax(self.pool_attn(seq), dim=1)
            return (seq * weights).sum(dim=1)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def _to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        # CSI-like: (B,3,114,T) -> (B,T,342)
        if x.ndim == 4 and x.size(1) == 3 and x.size(2) == 114:
            b = x.size(0)
            t = x.size(-1)
            return x.view(b, 3 * 114, t).permute(0, 2, 1)
        # UT-HAR: (B,1,250,90) -> (B,250,90)
        if x.ndim == 4 and x.size(1) == 1 and x.size(2) == 250:
            return x.squeeze(1)  # (B,250,90)
        # Widar: (B,22,20,20) -> (B,20,440)
        if x.ndim == 4 and x.size(1) == 22 and x.size(2) == 20 and x.size(3) == 20:
            b = x.size(0)
            x = x.permute(0, 2, 1, 3)  # (B,20,22,20)
            return x.reshape(b, 20, 22 * 20)
        raise ValueError(f"Unsupported input shape for GenericMambaClassifier: {tuple(x.shape)}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._to_sequence(x)
        seq = self.input_proj(seq)
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)
        return self._pool(seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats)

