"""Runtime inference wrapper used by the API and the CLI demo.

Loads the trained artifacts once and turns a raw multimodal session (exactly the
JSON the browser collector posts) into a risk decision with explanations.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .config import CONFIG, MODEL_DIR
from .features.aggregate import session_to_features, sessions_to_sequences
from .models.deep import BehaviorNet
from .models.baselines import proba_bot
from .models.risk import score as risk_score, RiskResult


class Detector:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.cfg = CONFIG
        ck = torch.load(MODEL_DIR / "deep.pt", map_location=device)
        self.seq_len = ck["seq_len"]
        self.net = BehaviorNet(in_ch=ck["in_ch"]).to(device)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.baselines = joblib.load(MODEL_DIR / "baselines.joblib")
        self.feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())

    def _sequence(self, session: dict) -> np.ndarray:
        Q, _ = sessions_to_sequences([session], self.seq_len)
        return Q

    @torch.no_grad()
    def score(self, session: dict, baseline: str = "xgboost") -> dict:
        Q = self._sequence(session)
        logits, attn = self.net(torch.tensor(Q).to(self.device), return_attn=True)
        p_deep = float(torch.softmax(logits, 1)[0, 1])
        attn = attn[0].cpu().numpy()

        # classical baseline probability (for the comparison view)
        feats = session_to_features(session)
        x = np.array([[feats[k] for k in self.feature_names]], float)
        x = np.nan_to_num(x)
        bl = self.baselines.get(baseline) or next(iter(self.baselines.values()))
        p_base = float(proba_bot(bl, x)[0])

        rr: RiskResult = risk_score(session, p_deep, self.cfg.risk)
        # top attended trajectory positions (fraction along the path)
        top_pos = (np.argsort(attn)[::-1][:5] / max(1, len(attn) - 1)).round(3).tolist()

        return {
            "risk": round(rr.risk, 4),
            "action": rr.action,
            "p_bot_deep": round(p_deep, 4),
            "p_bot_baseline": round(p_base, 4),
            "baseline_model": baseline,
            "reasons": rr.reasons,
            "attention_peaks_path_fraction": top_pos,
            "features": {k: round(float(v), 4) for k, v in feats.items()},
        }


_DETECTOR: Detector | None = None


def get_detector() -> Detector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector()
    return _DETECTOR
