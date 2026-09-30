"""Classical ML baselines on the aggregate feature vector.

These provide the 'traditional machine-learning' comparison required by the
objectives, and the tree models double as fast, explainable production
fallbacks. All are wrapped in a StandardScaler pipeline where it matters.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


def _to_binary(y):
    return (np.asarray(y) != 0).astype(int)


def build_baselines(seed: int = 1337) -> dict:
    models = {
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "svm_rbf": Pipeline([("sc", StandardScaler()),
                            ("clf", SVC(probability=True, class_weight="balanced", random_state=seed))]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=-1)
    return models


def fit_baselines(models: dict, Xtr, ytr):
    ytr_b = _to_binary(ytr)
    for name, m in models.items():
        if name == "xgboost":
            pos = max(1, int((ytr_b == 1).sum()))
            neg = max(1, int((ytr_b == 0).sum()))
            m.set_params(scale_pos_weight=neg / pos)
        m.fit(Xtr, ytr_b)
    return models


def proba_bot(model, X):
    return model.predict_proba(X)[:, 1]
