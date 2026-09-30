"""Render publication-quality figures for the paper from generated sessions.

Outputs to artifacts/reports/:
  fig_trajectories.png     example human vs bot mouse paths (grid)
  fig_separability.png     class-conditional feature distributions
These complement roc.png, confusion.png and shap_importance.png from training.

Run:  python scripts/make_paper_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcg.config import CONFIG, REPORT_DIR
from bcg.datagen import trajectories as T
from bcg.datagen.sessions import build_dataset
from bcg.features.aggregate import sessions_to_matrix

INK, PANEL, GRID, HUMAN, BOT, TEXT, MUTED = (
    "#0A0E14", "#0F141C", "#243040", "#35D0BA", "#F0A93B", "#C7D0DB", "#6B7686")
plt.rcParams.update({"text.color": TEXT, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9})


def fig_trajectories():
    rng = np.random.default_rng(3)
    fams = ["human", "selenium_linear", "puppeteer_step", "bezier_bot", "minjerk_adv"]
    titles = ["Human", "Selenium (linear)", "Puppeteer (step)", "Bezier bot", "Min-jerk adversary"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), facecolor=INK)
    for col, (fam, title) in enumerate(zip(fams, titles)):
        color = HUMAN if fam == "human" else BOT
        for row in range(2):
            ax = axes[row, col]
            tr = T.GENERATORS[fam](rng, CONFIG.data.screen_w, CONFIG.data.screen_h)
            x, y, t = tr[:, 0], tr[:, 1], tr[:, 2]
            # colour by speed to expose the velocity profile
            v = np.hypot(np.diff(x), np.diff(y)) / (np.diff(t) + 1e-6)
            v = v / (v.max() + 1e-9)
            for i in range(1, len(x)):
                ax.plot(x[i-1:i+1], y[i-1:i+1], color=color, alpha=0.35 + 0.65*v[i-1], lw=1.6)
            ax.scatter([x[0]], [y[0]], c=[MUTED], s=14, zorder=3)
            ax.scatter([x[-1]], [y[-1]], c=[color], s=22, zorder=3)
            ax.set_facecolor(PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if row == 0:
                ax.set_title(title, color=TEXT, fontsize=11, pad=8)
    fig.suptitle("Representative mouse trajectories by class (opacity ∝ instantaneous speed)",
                 color=TEXT, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(REPORT_DIR / "fig_trajectories.png", dpi=140, facecolor=INK)
    plt.close(fig)
    print("wrote", REPORT_DIR / "fig_trajectories.png")


def fig_separability():
    sessions = build_dataset(CONFIG)
    X, y, names, fams = sessions_to_matrix(sessions)
    fams = np.array(fams)
    feats = ["straightness", "speed_cv", "jerk_std", "dt_regularity"]
    labels = ["Path straightness", "Speed variability (CV)", "Jerk std", "Clock irregularity"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), facecolor=INK)
    for ax, f, lab in zip(axes, feats, labels):
        j = names.index(f)
        hvals = X[fams == "human", j]
        bvals = X[fams != "human", j]
        lo, hi = np.percentile(np.concatenate([hvals, bvals]), [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(hvals, bins=bins, color=HUMAN, alpha=0.6, label="human", density=True)
        ax.hist(bvals, bins=bins, color=BOT, alpha=0.6, label="bot", density=True)
        ax.set_facecolor(PANEL); ax.set_title(lab, color=TEXT, fontsize=10)
        ax.tick_params(labelsize=7)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0].legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    fig.suptitle("Class-conditional feature distributions (why naive bots separate and the adversary does not)",
                 color=TEXT, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(REPORT_DIR / "fig_separability.png", dpi=140, facecolor=INK)
    plt.close(fig)
    print("wrote", REPORT_DIR / "fig_separability.png")


if __name__ == "__main__":
    fig_trajectories()
    fig_separability()
