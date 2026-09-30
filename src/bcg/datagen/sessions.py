"""Assemble full multimodal interaction *sessions*.

A session bundles the four modalities the objectives call for:
  1. mouse movement trajectory
  2. keystroke dynamics
  3. click patterns (count + inter-click timing + press duration)
  4. session-level interaction metadata (durations, event rate, tab presence)

This is the canonical record consumed everywhere downstream (features, model,
API). Keeping one schema means the browser collector, the training data and the
live API all speak the same language.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from ..config import LABELS, CONFIG
from . import trajectories as T
from . import keystrokes as K


def _click_pattern(rng: np.random.Generator, human: bool) -> dict:
    n_clicks = int(rng.integers(1, 4))
    if human:
        inter = rng.lognormal(np.log(400), 0.6, max(1, n_clicks - 1))
        press = rng.lognormal(np.log(90), 0.35, n_clicks)   # button-down duration
    else:
        inter = np.full(max(1, n_clicks - 1), rng.uniform(50, 120)) + rng.normal(0, 4, max(1, n_clicks - 1))
        press = np.full(n_clicks, rng.uniform(10, 40)) + rng.normal(0, 2, n_clicks)
    return {
        "n_clicks": n_clicks,
        "inter_click_ms": np.clip(inter, 1, None).tolist(),
        "press_ms": np.clip(press, 1, None).tolist(),
    }


def make_session(rng: np.random.Generator, kind: str, cfg=CONFIG) -> dict[str, Any]:
    """Produce one labelled multimodal session dictionary."""
    human = kind == "human"
    # The advanced adversary mimics human timing on the *auxiliary* modalities
    # (keystroke/click/flags); its only residual tell is the mouse trajectory's
    # temporal shape. This forces the detector to rely on real behavioural
    # structure rather than easy environment flags.
    mimic = kind == "minjerk_adv"
    looks_human = human or mimic

    traj = T.GENERATORS[kind](rng, cfg.data.screen_w, cfg.data.screen_h)

    if looks_human:
        ks = K.gen_human_keystrokes(rng)
    else:
        ks = K.gen_bot_keystrokes(rng, kind)

    clicks = _click_pattern(rng, looks_human)

    total_ms = float(traj[-1, 2] - traj[0, 2]) + float(np.sum(ks["flight"])) + float(np.sum(clicks["inter_click_ms"]))
    n_events = len(traj) + len(ks["dwell"]) + clicks["n_clicks"]

    session = {
        "label": LABELS[kind],
        "family": kind,
        "mouse": {
            "x": traj[:, 0].tolist(),
            "y": traj[:, 1].tolist(),
            "t": traj[:, 2].tolist(),
        },
        "keystroke": {"dwell": ks["dwell"].tolist(), "flight": ks["flight"].tolist()},
        "clicks": clicks,
        "session": {
            "total_ms": total_ms,
            "n_events": int(n_events),
            "event_rate_hz": float(n_events / (total_ms / 1000 + 1e-6)),
            # coarse environment hints a real collector would also send:
            "has_focus_events": bool(looks_human or rng.random() < 0.2),
            "has_scroll": bool(looks_human and rng.random() < 0.7),
            "webdriver_flag": bool((not looks_human) and rng.random() < 0.5),
        },
    }
    return session


def build_dataset(cfg=CONFIG) -> list[dict]:
    rng = np.random.default_rng(cfg.data.seed)
    sessions: list[dict] = []
    for _ in range(cfg.data.n_human):
        sessions.append(make_session(rng, "human", cfg))
    for fam in ("selenium_linear", "puppeteer_step", "bezier_bot", "minjerk_adv"):
        for _ in range(cfg.data.n_per_bot):
            sessions.append(make_session(rng, fam, cfg))
    rng.shuffle(sessions)
    return sessions
