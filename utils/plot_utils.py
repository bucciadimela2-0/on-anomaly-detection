import os
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from utils.run_utils import ensure_dir


# Plot training curves (loss, r, etc.)
def plot_training_curves(
    history: Dict[str, list],
    out_dir: str,
    prefix: str = "ocnn",
) -> None:
    ensure_dir(out_dir)

    if "loss" in history and len(history["loss"]) > 0:
        plt.figure()
        plt.plot(np.arange(1, len(history["loss"]) + 1), history["loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{prefix} - training loss")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), dpi=200, bbox_inches="tight")
        plt.close()

    if "r" in history and len(history["r"]) > 0:
        plt.figure()
        plt.plot(np.arange(1, len(history["r"]) + 1), history["r"])
        plt.xlabel("Epoch")
        plt.ylabel("r")
        plt.title(f"{prefix} - r (nu-quantile)")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{prefix}_r.png"), dpi=200, bbox_inches="tight")
        plt.close()


# Histogram of anomaly scores for normal vs anomaly samples
def plot_score_histogram(
    anom_scores: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    filename: str = "score_hist.png",
    bins: int = 50,
    anomaly_label: int = 1,
    score_name: str = "anomaly score",
) -> None:
    ensure_dir(out_dir)

    s = anom_scores.detach().cpu().numpy().reshape(-1)
    y = labels.detach().cpu().numpy().reshape(-1)

    s_norm = s[y != anomaly_label]
    s_anom = s[y == anomaly_label]

    plt.figure()
    plt.hist(s_norm, bins=bins, alpha=0.7, label="Normal")
    plt.hist(s_anom, bins=bins, alpha=0.7, label="Anomaly")
    plt.xlabel(score_name)
    plt.ylabel("Count")
    plt.title("Score distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


# Boxplot of anomaly scores for normal vs anomaly samples
def plot_score_boxplot(
    anom_scores: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    filename: str = "score_boxplot.png",
    anomaly_label: int = 1,
    score_name: str = "anomaly score",
) -> None:
    ensure_dir(out_dir)

    s = anom_scores.detach().cpu().numpy().reshape(-1)
    y = labels.detach().cpu().numpy().reshape(-1)

    s_norm = s[y != anomaly_label]
    s_anom = s[y == anomaly_label]

    plt.figure()
    plt.boxplot(
        [s_norm, s_anom],
        labels=["Normal", "Anomaly"],
        showfliers=True,
    )
    plt.ylabel(score_name)
    plt.title("Score distribution (boxplot)")
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


def plot_extremes_in_class(
    images: torch.Tensor,
    anom_scores: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    filename: str = "extremes_in_class.png",
    target_label: int = 0,
    k: int = 8,
    score_name: str = "anomaly score",
) -> None:
    ensure_dir(out_dir)

    x = images.detach().cpu()  # [N,C,H,W] or [N,H,W]
    s = anom_scores.detach().cpu().numpy().reshape(-1)
    y = labels.detach().cpu().numpy().reshape(-1)

    idx = np.where(y == int(target_label))[0]
    if idx.size == 0:
        raise ValueError(f"No samples found with label={target_label}")

    k_eff = min(int(k), int(idx.size))
    s_sub = s[idx]
    order = np.argsort(s_sub)

    least_idx = idx[order[:k_eff]]
    most_idx = idx[order[-k_eff:]][::-1]

    def _imshow_auto(ax, img_t: torch.Tensor) -> None:
        """
        Supports:
          - grayscale: [H,W] or [1,H,W]
          - rgb:       [3,H,W]
        """
        img_t = img_t.detach().cpu()

        if img_t.ndim == 2:
            ax.imshow(img_t, cmap="gray")
            return

        if img_t.ndim == 3:
            # CHW
            c = img_t.shape[0]
            if c == 1:
                ax.imshow(img_t[0], cmap="gray")
                return
            if c == 3:
                ax.imshow(img_t.permute(1, 2, 0))  # HWC
                return

            # fallback: show first channel as gray
            ax.imshow(img_t[0], cmap="gray")
            return

        raise TypeError(f"Invalid shape {tuple(img_t.shape)} for image data")

    plt.figure(figsize=(1.6 * k_eff, 3.5))

    # least anomalous
    for j, i in enumerate(least_idx):
        ax = plt.subplot(2, k_eff, j + 1)
        img = x[i]
        _imshow_auto(ax, img)
        ax.set_title(f"{s[i]:.3f}")
        ax.axis("off")

    # most anomalous
    for j, i in enumerate(most_idx):
        ax = plt.subplot(2, k_eff, k_eff + j + 1)
        img = x[i]
        _imshow_auto(ax, img)
        ax.set_title(f"{s[i]:.3f}")
        ax.axis("off")

    plt.suptitle(
        f"Label={target_label} | least vs most anomalous ({score_name})",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close()


# Plot ROC curve and return AUROC
def plot_roc_curve(
    anom_scores: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    filename: str = "roc_curve.png",
    anomaly_label: int = 1,
) -> float:
    ensure_dir(out_dir)

    s = anom_scores.detach().cpu().numpy().reshape(-1)
    y = labels.detach().cpu().numpy().reshape(-1)

    y_bin = (y == anomaly_label).astype(np.int32)

    fpr, tpr, _ = roc_curve(y_bin, s)
    auc = roc_auc_score(y_bin, s)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUROC = {auc:.4f})")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close()

    return float(auc)


# Plot Precision–Recall curve and return Average Precision
def plot_pr_curve(
    anom_scores: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    filename: str = "pr_curve.png",
    anomaly_label: int = 1,
) -> float:
    ensure_dir(out_dir)

    s = anom_scores.detach().cpu().numpy().reshape(-1)
    y = labels.detach().cpu().numpy().reshape(-1)

    y_bin = (y == anomaly_label).astype(np.int32)

    precision, recall, _ = precision_recall_curve(y_bin, s)
    ap = average_precision_score(y_bin, s)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall curve (AP = {ap:.4f})")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close()

    return float(ap)
