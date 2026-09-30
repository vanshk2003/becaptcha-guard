"""Central configuration for BeCAPTCHA-Guard.

All tunable constants live here so experiments are reproducible and the
whole pipeline can be driven from one place (train, evaluate, serve).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data"
MODEL_DIR = ARTIFACTS / "models"
REPORT_DIR = ARTIFACTS / "reports"
for _d in (ARTIFACTS, DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Class taxonomy
#   0 = genuine human
#   1..N = bot families (all collapse to "bot" for the binary detector, but
#          multi-class labels are kept for per-family analysis / forensics)
# ---------------------------------------------------------------------------
LABELS = {
    "human": 0,
    "selenium_linear": 1,   # naive automation: teleport / linear interpolation
    "puppeteer_step": 2,    # fixed-step grid movement
    "bezier_bot": 3,        # "human-like" smooth Bezier libraries
    "minjerk_adv": 4,       # advanced adversary: minimum-jerk + tremor injection
}
BOT_LABELS = [v for k, v in LABELS.items() if k != "human"]
ID2NAME = {v: k for k, v in LABELS.items()}


@dataclass
class DataConfig:
    n_human: int = 1200
    n_per_bot: int = 320           # per bot family
    seq_len: int = 128             # resampled trajectory length for the DL model
    screen_w: int = 1280
    screen_h: int = 800
    sample_hz: int = 60            # nominal event rate
    seed: int = 1337


@dataclass
class TrainConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    batch_size: int = 64
    epochs: int = 25
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    seed: int = 1337
    device: str = "cpu"


@dataclass
class RiskConfig:
    """Adaptive, tiered risk policy (BOTracle-style layered response)."""
    allow_below: float = 0.35      # p(bot) < this -> ALLOW silently
    challenge_below: float = 0.75  # this range -> step-up CHALLENGE
    # >= challenge_below -> BLOCK
    # Heuristic hard-stops override the model (defence in depth):
    max_straightness_human: float = 0.985   # near-perfect straight line = suspicious
    min_velocity_cv: float = 0.06           # near-constant velocity = suspicious
    min_events: int = 8                     # too few events to trust -> challenge


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


CONFIG = Config()
