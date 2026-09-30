"""End-to-end training: data -> features -> models -> evaluation -> artifacts.

Run:  python -m bcg.train
Produces in artifacts/:
  models/deep.pt, models/baselines.joblib, models/feature_names.json, config.json
  reports/metrics.json, reports/roc.png, reports/confusion.png, reports/comparison.md
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .config import CONFIG, MODEL_DIR, REPORT_DIR, ARTIFACTS
from .datagen.sessions import build_dataset
from .features.aggregate import sessions_to_matrix, sessions_to_sequences
from .models.baselines import build_baselines, fit_baselines, proba_bot
from .models.deep import train_deep, predict_proba_deep
from . import evaluate as ev


def main():
    cfg = CONFIG
    np.random.seed(cfg.train.seed)
    print("[1/6] generating multimodal dataset ...")
    sessions = build_dataset(cfg)
    print(f"      {len(sessions)} sessions "
          f"({sum(s['label']==0 for s in sessions)} human / "
          f"{sum(s['label']!=0 for s in sessions)} bot)")

    # split at the session level so no leakage between feature & sequence views
    idx = np.arange(len(sessions))
    y_all = np.array([s["label"] for s in sessions])
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.train.test_size,
                                      stratify=y_all, random_state=cfg.train.seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=cfg.train.val_size,
                                      stratify=y_all[tr_idx], random_state=cfg.train.seed)
    S = np.array(sessions, dtype=object)
    tr, va, te = list(S[tr_idx]), list(S[va_idx]), list(S[te_idx])

    print("[2/6] extracting features ...")
    Xtr, ytr, names, _ = sessions_to_matrix(tr)
    Xva, yva, _, _ = sessions_to_matrix(va)
    Xte, yte, _, fam_te = sessions_to_matrix(te)
    Qtr, ytr2 = sessions_to_sequences(tr, cfg.data.seq_len)
    Qva, yva2 = sessions_to_sequences(va, cfg.data.seq_len)
    Qte, yte2 = sessions_to_sequences(te, cfg.data.seq_len)
    yte_bin = (yte != 0).astype(int)

    print("[3/6] training classical baselines ...")
    baselines = fit_baselines(build_baselines(cfg.train.seed), Xtr, ytr)

    print("[4/6] training deep BehaviorNet ...")
    model, best_va = train_deep(Qtr, ytr2, Qva, yva2, cfg.train, device=cfg.train.device)
    print(f"      best val acc = {best_va:.3f}")

    print("[5/6] evaluating on held-out test set ...")
    metrics, roc_curves = {}, {}
    for name, m in baselines.items():
        p = proba_bot(m, Xte)
        metrics[name] = ev.binary_metrics(yte_bin, p)
        metrics[name].update(ev.per_family_recall(yte, p, fam_te))
        roc_curves[name] = (yte_bin, p)

    p_deep = predict_proba_deep(model, Qte, device=cfg.train.device)
    metrics["deep_behaviornet"] = ev.binary_metrics(yte_bin, p_deep)
    metrics["deep_behaviornet"].update(ev.per_family_recall(yte, p_deep, fam_te))
    roc_curves["deep_behaviornet"] = (yte_bin, p_deep)

    print("[6/6] saving models, reports and plots ...")
    torch.save({"state_dict": model.state_dict(), "in_ch": Qtr.shape[2],
                "seq_len": cfg.data.seq_len}, MODEL_DIR / "deep.pt")
    joblib.dump(baselines, MODEL_DIR / "baselines.joblib")
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(names, indent=2))
    cfg.save(ARTIFACTS / "config.json")

    ev.save_report(metrics, REPORT_DIR / "metrics.json")
    ev.plot_roc(roc_curves, REPORT_DIR / "roc.png")
    ev.plot_confusion(yte_bin, p_deep, REPORT_DIR / "confusion.png")
    _write_comparison(metrics, REPORT_DIR / "comparison.md")

    print("\n=== TEST-SET SUMMARY (p(bot) >= 0.5) ===")
    hdr = f"{'model':<20}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'auc':>7}{'eer':>7}"
    print(hdr); print("-" * len(hdr))
    for name, mm in metrics.items():
        print(f"{name:<20}{mm['accuracy']:>7.3f}{mm['precision']:>7.3f}"
              f"{mm['recall']:>7.3f}{mm['f1']:>7.3f}{mm['roc_auc']:>7.3f}{mm['eer']:>7.3f}")
    print("\nartifacts written to", ARTIFACTS)


def _write_comparison(metrics: dict, path: Path):
    lines = ["# Model comparison (held-out test set)\n",
             "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | EER |",
             "|---|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
                     f"{m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['eer']:.3f} |")
    lines.append("\n## Per-family recall (did we catch each bot type?)\n")
    fam_keys = [k for k in next(iter(metrics.values())) if k.endswith("_recall") or k.endswith("_TNR")]
    lines.append("| Model | " + " | ".join(fam_keys) + " |")
    lines.append("|---|" + "|".join(["---"] * len(fam_keys)) + "|")
    for name, m in metrics.items():
        lines.append(f"| {name} | " + " | ".join(f"{m.get(k, float('nan')):.3f}" for k in fam_keys) + " |")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
