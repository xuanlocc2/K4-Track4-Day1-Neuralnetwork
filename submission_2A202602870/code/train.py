"""train.py — seed, đánh giá, run_experiment(cfg, data), dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là đổi dict cfg rồi gọi lại run_experiment.
"""
from __future__ import annotations

import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """cm hàng = nhãn thật, cột = dự đoán. F1_c = 2TP/(số dự đoán lớp c + số thật lớp c), 0 nếu mẫu = 0."""
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0).mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str):
    """"ce": cross-entropy trên logit thô. "mse": MSE giữa logit và one-hot, trung bình trên MỌI phần tử (N*7)."""
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        return F.mse_loss(logits.float(), F.one_hot(y, logits.shape[1]).float())
    raise ValueError(loss_name)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1) ở eval() (dropout tắt). Loss = tổng loss từng mẫu / N."""
    model.eval()
    tot, cm = 0.0, torch.zeros(49, dtype=torch.long, device=X.device)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        tot += compute_loss(logits, yb, loss_name).item() * len(yb)
        cm += torch.bincount(yb * 7 + logits.argmax(1), minlength=49)
    cm = cm.view(7, 7).cpu().numpy()
    return dict(loss=tot / len(X), acc=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1_from_confusion(cm))


def run_experiment(cfg: dict, data: dict) -> dict:
    cfg = {**DEFAULT_CFG, **cfg}
    assert cfg["lr"] is not None, "cfg['lr'] chưa đặt"
    set_seed(cfg["seed"])
    Xtr, ytr, Xval, yval = (data[k] for k in ("X_tr", "y_tr", "X_val", "y_val"))
    dev = Xtr.device
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden, cfg["dropout"], cfg["init"]).to(dev)
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden]
    opt = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"], cfg["momentum"])
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg["precision"])
    amp = dtype is not None
    scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "fp16" and dev.type == "cuda" else None
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"])
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    # tập con CỐ ĐỊNH (không phụ thuộc seed của cfg) để đo train_loss ở eval()
    sub = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(0))[:50_000].to(dev)
    Xsub, ysub = Xtr[sub], ytr[sub]

    h = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s")}
    step0 = evaluate(model, Xval, yval, cfg["loss"])["loss"]   # trước bước cập nhật đầu tiên
    best_val, best_epoch, best_state, diverged = math.inf, 0, None, False
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        sync()
        t0, gns = time.time(), []
        for xb, yb in iterate_batches(Xtr, ytr, cfg["batch"], gen):
            with torch.autocast(dev.type, dtype=dtype, enabled=amp):
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss):
                diverged = True
                break
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)          # phải unscale TRƯỚC khi clip
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            if math.isfinite(gn):             # fp16: gn = inf nghĩa là scaler bỏ qua bước này
                gns.append(gn)
        if diverged:
            break
        sync()
        dt = time.time() - t0
        tr, va = evaluate(model, Xsub, ysub, cfg["loss"]), evaluate(model, Xval, yval, cfg["loss"])
        if not math.isfinite(va["loss"]):
            diverged = True
            break
        for k, v in zip(h, (ep, tr["loss"], va["loss"], va["acc"], va["macro_f1"],
                            float(np.mean(gns)) if gns else float("nan"), dt)):
            h[k].append(v)
        if va["loss"] < best_val:
            best_val, best_epoch = va["loss"], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    nan = float("nan")
    if best_epoch:
        b = best_epoch - 1
        s = dict(best_val_loss=best_val, best_epoch=best_epoch, final_train_loss=h["train_loss"][-1],
                 final_val_loss=h["val_loss"][-1], val_acc=h["val_acc"][b], val_macro_f1=h["val_macro_f1"][b],
                 time_per_epoch_s=float(np.mean(h["epoch_time_s"])))
    else:   # phân kỳ ngay epoch 1
        s = dict(best_val_loss=nan, best_epoch=0, final_train_loss=nan, final_val_loss=nan,
                 val_acc=nan, val_macro_f1=nan, time_per_epoch_s=nan)
    s.update(step0_loss=step0, diverged=diverged,
             peak_mem_MB=torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0)
    return {"cfg": cfg, "history": h, "summary": s, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    df = pd.DataFrame({"row_id": np.asarray(row_id), "pred": np.asarray(preds)})
    assert df["row_id"].is_unique and df["pred"].between(0, 6).all()
    df.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> str:
    """Dùng MỘT LẦN cho cấu hình cuối (và baseline): nạp best_state, dự đoán eval, ghi CSV, chạy evaluate.py."""
    cfg = {**DEFAULT_CFG, **cfg}
    model = MLP(tuple(cfg["hidden"]), cfg["dropout"], cfg["init"]).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    write_predictions(data["eval_row_id"], predict(model, data["X_eval"]).cpu().numpy(), pred_path)
    root = Path(__file__).resolve().parents[2]    # code/ -> submission_<MSSV>/ -> gốc repo
    out = subprocess.run([sys.executable, "scripts/evaluate.py", "--pred", str(Path(pred_path).resolve())],
                         cwd=root, capture_output=True, text=True)
    print(out.stdout, out.stderr)
    return out.stdout
