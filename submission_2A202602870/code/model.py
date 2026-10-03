"""model.py — MLP cho bài toán 7 lớp.

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> ... -> Linear(h_last, 7) -> logits (B, 7)

Lớp cuối ra logit thô (không softmax); Dropout chỉ sau ReLU của lớp ẩn; mọi Linear có bias;
không BatchNorm, không residual. Số tham số phải khớp EXPECTED_PARAMS.
"""
from __future__ import annotations

import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base
    (512, 256): 161_287,       # M-wide
    (256, 128, 64): 55_687,    # M-deep
}


class MLP(nn.Module):
    """hidden: tuple số nơ-ron lớp ẩn; dropout: xác suất tắt q; init: zeros|normal|xavier|he|default."""

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers, d = [], in_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo mọi nn.Linear (bias luôn = 0). xavier = nn.init.xavier_normal_ (Var = 2/(n_in+n_out))."""
    assert init in ("zeros", "normal", "xavier", "he", "default"), init
    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        if init == "zeros":
            nn.init.zeros_(m.weight)
        elif init == "normal":
            nn.init.normal_(m.weight, 0.0, 0.01)
        elif init == "xavier":
            nn.init.xavier_normal_(m.weight)
        elif init == "he":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if init != "default":
            nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của đầu ra sau mỗi nn.Linear (trước ReLU), eval mode, một lô x."""
    model.eval()
    h, stds = x, []
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Linear):
            stds.append(h.std().item())
    return stds
