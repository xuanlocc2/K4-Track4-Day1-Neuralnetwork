"""plots.py — ảnh từng thí nghiệm và ảnh chồng."""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    c, h, s = result["cfg"], result["history"], result["summary"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train (eval mode)")
    ax[0].plot(h["epoch"], h["val_loss"], label="val")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(h["epoch"], h["val_acc"], label="val acc")
    ax[1].plot(h["epoch"], h["val_macro_f1"], label="val macro-F1")
    ax[1].set(xlabel="epoch", ylabel="score", title="Val accuracy / macro-F1")
    ax[2].plot(h["epoch"], h["grad_norm"], label="grad norm (trước clip)")
    ax[2].set(xlabel="epoch", ylabel="||g||₂", title="Grad norm")
    for a in ax:
        if s["best_epoch"]:
            a.axvline(s["best_epoch"], color="gray", ls="--", label="best epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.suptitle(f'{c["exp_id"]} | {c["optimizer"]} lr={c["lr"]} wd={c["weight_decay"]} batch={c["batch"]} '
                 f'hidden={tuple(c["hidden"])} drop={c["dropout"]} init={c["init"]} clip={c["clip_norm"]} '
                 f'{c["precision"]} loss={c["loss"]} seed={c["seed"]}', fontsize=9)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        h = r["history"]
        ax.plot(h["epoch"], h[metric], label=r["cfg"]["exp_id"])
    ax.set(xlabel="epoch", ylabel=metric, title=title or metric)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
