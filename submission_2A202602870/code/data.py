"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Quy ước: X float32 (N, 54), 10 cột đầu là số liên tục, 44 cột sau nhị phân; y int64 (N,) nhãn 0..6.
Tập eval CHỈ dùng để chấm điểm cuối, không dùng để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id"""
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    Xtr, ytr, Xev, yev, rid = tr["X"], tr["y"], ev["X"], ev["y"], ev["row_id"]
    assert Xtr.shape[1] == Xev.shape[1] == 54 and Xtr.dtype == np.float32
    assert ytr.dtype == np.int64 and ytr.min() == 0 and ytr.max() == 6
    return Xtr, ytr, Xev, yev, rid


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train, phân tầng theo nhãn. Trả về: X_tr, y_tr, X_val, y_val"""
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean/std của N_NUMERIC cột đầu, CHỈ trên train (sau khi tách val)."""
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    std = np.where(std == 0, 1.0, std)  # tránh chia cho 0
    return mean, std


def apply_standardizer(X, mean, std):
    """Bản sao của X với 10 cột đầu được chuẩn hoá; 44 cột nhị phân giữ nguyên."""
    X = X.copy()
    X[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std
    return X


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Trả về dict tensor trên device: X_tr, y_tr, X_val, y_val, X_eval, y_eval và eval_row_id (numpy)."""
    Xtr_full, ytr_full, X_eval, y_eval, rid = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(Xtr_full, ytr_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr, X_val, X_eval = (apply_standardizer(a, mean, std) for a in (X_tr, X_val, X_eval))
    t = lambda a: torch.tensor(a, device=device)
    d = dict(X_tr=t(X_tr).float(), y_tr=t(y_tr).long(),
             X_val=t(X_val).float(), y_val=t(y_val).long(),
             X_eval=t(X_eval).float(), y_eval=t(y_eval).long(), eval_row_id=rid)
    maj = np.bincount(y_tr).argmax()
    print("X_tr", tuple(d["X_tr"].shape), "X_val", tuple(d["X_val"].shape), "X_eval", tuple(d["X_eval"].shape))
    print(f"đoán lớp đa số trên val: {(y_val == maj).mean():.4f}")
    return d


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader. Batch cuối có thể nhỏ hơn batch_size (giữ lại)."""
    N = len(X)
    perm = torch.randperm(N, generator=generator, device=X.device) if shuffle else torch.arange(N, device=X.device)
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
