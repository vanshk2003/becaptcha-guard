"""Adaptive, layered risk-scoring and response engine.

Layered defence in the spirit of BOTracle (ESORICS 2024):
  Layer 1  heuristic hard-signals (webdriver flag, impossibly straight path,
           near-constant velocity, too few events) — cheap, high-precision.
  Layer 2  the learned model's p(bot).
  Layer 3  fusion into a single risk score and a tiered ACTION so that the
           experience stays seamless for genuine users (ALLOW) while raising
           friction only when warranted (CHALLENGE) and blocking clear bots.

The action, not just the score, is the product: this is what keeps false
positives from turning into blocked real users.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import RiskConfig
from ..features.mouse import mouse_features
import numpy as np


@dataclass
class RiskResult:
    risk: float                 # fused p(bot)-like score in [0,1]
    action: str                 # ALLOW | CHALLENGE | BLOCK
    model_p: float
    reasons: list[str]


def _heuristic_flags(session: dict, cfg: RiskConfig) -> tuple[float, list[str]]:
    reasons: list[str] = []
    bonus = 0.0
    m = session["mouse"]
    traj = np.column_stack([m["x"], m["y"], m["t"]]).astype(float)
    feats = mouse_features(traj)

    if session.get("session", {}).get("webdriver_flag"):
        reasons.append("navigator.webdriver flag present")
        bonus = max(bonus, 0.9)
    if feats["straightness"] >= cfg.max_straightness_human:
        reasons.append(f"path near-perfectly straight ({feats['straightness']:.3f})")
        bonus = max(bonus, 0.7)
    if feats["speed_cv"] <= cfg.min_velocity_cv:
        reasons.append(f"velocity almost constant (cv={feats['speed_cv']:.3f})")
        bonus = max(bonus, 0.6)
    if feats["dt_regularity"] <= 0.01:
        reasons.append("timing on a perfectly regular clock")
        bonus = max(bonus, 0.5)
    if session.get("session", {}).get("n_events", 999) < cfg.min_events:
        reasons.append("too few interaction events to trust")
        bonus = max(bonus, 0.4)
    return bonus, reasons


def score(session: dict, model_p: float, cfg: RiskConfig) -> RiskResult:
    """Fuse model probability with heuristics into a risk + action."""
    bonus, reasons = _heuristic_flags(session, cfg)
    # fusion: heuristics can only *raise* risk (defence in depth, high precision)
    risk = float(max(model_p, 0.5 * model_p + 0.5 * bonus if bonus else model_p))
    risk = min(1.0, max(0.0, risk))

    if risk < cfg.allow_below:
        action = "ALLOW"
    elif risk < cfg.challenge_below:
        action = "CHALLENGE"
    else:
        action = "BLOCK"

    if not reasons:
        reasons = ["no hard heuristic triggers; decision from behavioural model"]
    return RiskResult(risk=risk, action=action, model_p=float(model_p), reasons=reasons)
