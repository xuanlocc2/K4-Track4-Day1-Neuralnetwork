"""optimizer.py — chọn bộ tối ưu và cắt gradient (torch.optim, clip_grad_norm_)."""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":   # weight_decay = L2 trộn vào gradient
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)  # suy giảm tách riêng


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Tuỳ chọn. Hiện chỉ hỗ trợ "cosine"; train.py chưa gọi (thêm khi cần)."""
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(name)


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt theo chuẩn L2 toàn cục; trả về chuẩn TRƯỚC khi cắt (max_norm=None -> chỉ đo)."""
    total = torch.nn.utils.clip_grad_norm_(params, float("inf") if max_norm is None else max_norm)
    return float(total)
