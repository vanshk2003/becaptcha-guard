"""Explainability.

Two complementary views satisfy the 'explainable' objective:
  * Feature level  — SHAP TreeExplainer over the XGBoost/RF baseline gives global
    and per-decision attributions in terms of the interpretable kinematic
    features (which behaviours drove the risk).
  * Sequence level — the deep model's attention weights (exposed by the API)
    show *where in the trajectory* the model focused.

Run:  python -m bcg.explain.shap_explain
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import MODEL_DIR, REPORT_DIR
from ..datagen.sessions import build_dataset
from ..features.aggregate import sessions_to_matrix


def run(max_samples: int = 400):
    import shap
    baselines = joblib.load(MODEL_DIR / "baselines.joblib")
    model = baselines.get("xgboost") or baselines["random_forest"]
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())

    sessions = build_dataset()
    X, y, fnames, _ = sessions_to_matrix(sessions)
    idx = np.random.default_rng(0).choice(len(X), size=min(max_samples, len(X)), replace=False)
    Xs = X[idx]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):          # some versions return per-class list
        sv = sv[1]
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]

    top = [{"feature": fnames[i], "mean_abs_shap": float(mean_abs[i])} for i in order[:15]]
    (REPORT_DIR / "shap_top_features.json").write_text(json.dumps(top, indent=2))

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0A0E14")
    ax.set_facecolor("#0A0E14")
    yy = np.arange(len(top))[::-1]
    ax.barh(yy, [t["mean_abs_shap"] for t in top], color="#35D0BA")
    ax.set_yticks(yy); ax.set_yticklabels([t["feature"] for t in top], color="#C7D0DB", fontsize=8)
    ax.set_xlabel("mean |SHAP| (impact on p(bot))", color="#C7D0DB")
    ax.set_title("Feature importance — behavioural drivers of bot risk", color="#C7D0DB")
    ax.tick_params(colors="#6B7686")
    for s in ax.spines.values():
        s.set_color("#243040")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "shap_importance.png", dpi=130); plt.close(fig)
    print("top drivers:", ", ".join(t["feature"] for t in top[:6]))
    print("written:", REPORT_DIR / "shap_importance.png")


if __name__ == "__main__":
    run()
