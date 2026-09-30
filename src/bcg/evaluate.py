"""Evaluation metrics and plotting.

Reports the full slate the objectives ask for: accuracy, precision, recall, F1,
ROC-AUC and Equal Error Rate (EER, the standard biometric operating point), a
confusion matrix, and — crucially for this problem — *per-bot-family recall*,
which reveals whether the detector actually catches the hard, human-like
adversary or just the naive bots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

from .config import ID2NAME

# oscilloscope-style palette to match the dashboard
C_HUMAN, C_BOT, C_INK, C_GRID = "#35D0BA", "#F0A93B", "#0A0E14", "#243040"


def equal_error_rate(y_true, p) -> float:
    fpr, tpr, _ = roc_curve(y_true, p)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def binary_metrics(y_true_bin, p, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_bin, pred)),
        "precision": float(precision_score(y_true_bin, pred, zero_division=0)),
        "recall": float(recall_score(y_true_bin, pred, zero_division=0)),
        "f1": float(f1_score(y_true_bin, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_bin, p)) if len(np.unique(y_true_bin)) > 1 else float("nan"),
        "eer": equal_error_rate(y_true_bin, p),
    }


def per_family_recall(y_multi, p, fams, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    out = {}
    for fam in sorted(set(fams)):
        if fam == "human":
            mask = np.array(fams) == "human"
            out["human_TNR"] = float((pred[mask] == 0).mean()) if mask.any() else float("nan")
        else:
            mask = np.array(fams) == fam
            out[f"{fam}_recall"] = float((pred[mask] == 1).mean()) if mask.any() else float("nan")
    return out


def plot_roc(curves: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=C_INK)
    ax.set_facecolor(C_INK)
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=C_GRID, lw=1)
    ax.set_xlabel("False positive rate (humans blocked)", color="#C7D0DB")
    ax.set_ylabel("True positive rate (bots caught)", color="#C7D0DB")
    ax.set_title("ROC — human vs. bot detection", color="#C7D0DB")
    ax.tick_params(colors="#6B7686")
    for s in ax.spines.values():
        s.set_color(C_GRID)
    ax.legend(facecolor="#121821", edgecolor=C_GRID, labelcolor="#C7D0DB", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(y_bin, p, path: Path, thr: float = 0.5):
    cm = confusion_matrix(y_bin, (p >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor=C_INK)
    ax.imshow(cm, cmap="cividis")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="#0A0E14", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["human", "bot"], color="#C7D0DB")
    ax.set_yticklabels(["human", "bot"], color="#C7D0DB")
    ax.set_xlabel("predicted", color="#C7D0DB"); ax.set_ylabel("actual", color="#C7D0DB")
    ax.set_title("Confusion matrix (deep model)", color="#C7D0DB")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_report(report: dict, path: Path):
    Path(path).write_text(json.dumps(report, indent=2))
