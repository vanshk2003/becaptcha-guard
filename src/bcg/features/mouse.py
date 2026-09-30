"""Mouse kinematic feature extraction.

Two outputs are produced from a trajectory:
  * `mouse_features(traj)` -> flat dict of interpretable kinematic descriptors
    (used by the classical ML baselines and by SHAP explainability).
  * `resample_sequence(traj, L)` -> (L, 4) array of [dx, dy, dt, speed] used as
    the input sequence for the deep 1D-CNN/BiLSTM model.

The descriptors target the physical differences between human and machine
motion: humans have variable, bell-shaped velocity, non-zero jerk, curvature
and corrective sub-movements; naive bots are straight and constant-velocity.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray
EPS = 1e-9


def _derivatives(x, y, t):
    dt = np.diff(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    dx, dy = np.diff(x), np.diff(y)
    vx, vy = dx / dt, dy / dt
    speed = np.hypot(vx, vy)
    return dt, dx, dy, vx, vy, speed


def _stats(a: Array, prefix: str) -> dict:
    if len(a) == 0:
        a = np.array([0.0])
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_std": float(np.std(a)),
        f"{prefix}_max": float(np.max(a)),
        f"{prefix}_min": float(np.min(a)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_cv": float(np.std(a) / (abs(np.mean(a)) + EPS)),
    }


def mouse_features(traj: Array) -> dict:
    x, y, t = traj[:, 0], traj[:, 1], traj[:, 2]
    n = len(x)
    if n < 3:
        # degenerate: return zeros with the right keys via a dummy path
        traj = np.array([[0, 0, 0], [1, 1, 10], [2, 2, 20]], float)
        x, y, t = traj[:, 0], traj[:, 1], traj[:, 2]

    dt, dx, dy, vx, vy, speed = _derivatives(x, y, t)
    accel = np.diff(speed) / (dt[:-1] + EPS) if len(speed) > 1 else np.array([0.0])
    jerk = np.diff(accel) / (dt[:-2] + EPS) if len(accel) > 1 else np.array([0.0])

    path_len = float(np.sum(np.hypot(dx, dy)))
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    straightness = disp / (path_len + EPS)          # 1.0 => perfect straight line

    # turning angles / curvature
    ang = np.arctan2(dy, dx)
    dang = np.diff(np.unwrap(ang)) if len(ang) > 1 else np.array([0.0])
    curvature = np.abs(dang) / (np.hypot(dx[1:], dy[1:]) + EPS) if len(dx) > 1 else np.array([0.0])

    # pauses and direction reversals
    pause_ratio = float(np.mean(speed < (0.05 * (np.mean(speed) + EPS))))
    reversals = int(np.sum(np.abs(dang) > (np.pi / 2))) if len(dang) else 0

    feats: dict = {
        "n_points": float(n),
        "duration_ms": float(t[-1] - t[0]),
        "path_length": path_len,
        "displacement": disp,
        "straightness": straightness,
        "pause_ratio": pause_ratio,
        "reversals": float(reversals),
        "dt_regularity": float(np.std(dt) / (np.mean(dt) + EPS)),  # low => machine clock
    }
    feats.update(_stats(speed, "speed"))
    feats.update(_stats(accel, "accel"))
    feats.update(_stats(jerk, "jerk"))
    feats.update(_stats(curvature, "curv"))
    feats.update(_stats(dang, "angchg"))
    return feats


def resample_sequence(traj: Array, length: int) -> Array:
    """Resample to fixed length; channels = [dx, dy, dt, speed], per-sample z-ish scaled."""
    x, y, t = traj[:, 0], traj[:, 1], traj[:, 2]
    if len(x) < 2:
        return np.zeros((length, 4), np.float32)
    # arc-length parameterisation preserves shape better than raw index
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        s = np.arange(len(x), dtype=float)
    u = np.linspace(0, s[-1], length + 1)
    xi = np.interp(u, s, x)
    yi = np.interp(u, s, y)
    ti = np.interp(u, s, t)
    dx = np.diff(xi)
    dy = np.diff(yi)
    dtt = np.diff(ti)
    dtt[dtt <= 0] = np.median(dtt[dtt > 0]) if np.any(dtt > 0) else 1.0
    speed = np.hypot(dx, dy) / dtt
    seqs = np.stack([dx, dy, dtt, speed], axis=1).astype(np.float32)
    # robust scaling so magnitudes across sessions are comparable
    for c in range(seqs.shape[1]):
        col = seqs[:, c]
        sd = np.std(col) + EPS
        seqs[:, c] = (col - np.mean(col)) / sd
    return seqs
