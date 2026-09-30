"""Keystroke / click / session features and the top-level aggregator.

`session_to_features(session)` returns a single flat, ordered dict combining all
modalities. `sessions_to_matrix(sessions)` vectorises a list into (X, y, names)
for the classical models, and `sessions_to_sequences` builds the tensor for the
deep model.
"""
from __future__ import annotations

import numpy as np

from .mouse import mouse_features, resample_sequence

EPS = 1e-9


def _stats(a, prefix):
    a = np.asarray(a, float)
    if a.size == 0:
        a = np.array([0.0])
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_std": float(np.std(a)),
        f"{prefix}_cv": float(np.std(a) / (abs(np.mean(a)) + EPS)),
        f"{prefix}_min": float(np.min(a)),
        f"{prefix}_max": float(np.max(a)),
    }


def keystroke_features(ks: dict) -> dict:
    f = {}
    f.update(_stats(ks.get("dwell", []), "dwell"))
    f.update(_stats(ks.get("flight", []), "flight"))
    dwell = np.asarray(ks.get("dwell", []), float)
    f["kd_n"] = float(dwell.size)
    return f


def click_features(clicks: dict) -> dict:
    f = {"n_clicks": float(clicks.get("n_clicks", 0))}
    f.update(_stats(clicks.get("inter_click_ms", []), "interclick"))
    f.update(_stats(clicks.get("press_ms", []), "press"))
    return f


def session_features(sess: dict) -> dict:
    return {
        "sess_total_ms": float(sess.get("total_ms", 0)),
        "sess_n_events": float(sess.get("n_events", 0)),
        "sess_event_rate": float(sess.get("event_rate_hz", 0)),
        "sess_focus": float(bool(sess.get("has_focus_events", False))),
        "sess_scroll": float(bool(sess.get("has_scroll", False))),
        "sess_webdriver": float(bool(sess.get("webdriver_flag", False))),
    }


def session_to_features(session: dict) -> dict:
    m = session["mouse"]
    traj = np.column_stack([m["x"], m["y"], m["t"]]).astype(float)
    feats = {}
    feats.update(mouse_features(traj))
    feats.update(keystroke_features(session.get("keystroke", {})))
    feats.update(click_features(session.get("clicks", {})))
    feats.update(session_features(session.get("session", {})))
    return feats


# ---------------------------------------------------------------------------
# vectorisers
# ---------------------------------------------------------------------------
def sessions_to_matrix(sessions):
    rows, y, fams = [], [], []
    for s in sessions:
        rows.append(session_to_features(s))
        y.append(int(s.get("label", 0)))
        fams.append(s.get("family", "unknown"))
    names = list(rows[0].keys())
    X = np.array([[r[k] for k in names] for r in rows], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(y), names, fams


def sessions_to_sequences(sessions, seq_len):
    seqs, y = [], []
    for s in sessions:
        m = s["mouse"]
        traj = np.column_stack([m["x"], m["y"], m["t"]]).astype(float)
        seqs.append(resample_sequence(traj, seq_len))
        y.append(int(s.get("label", 0)))   # unlabeled at inference time -> 0
    return np.stack(seqs).astype(np.float32), np.array(y)
