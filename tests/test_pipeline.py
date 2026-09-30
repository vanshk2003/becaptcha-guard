"""Fast sanity tests for the core pipeline (no training required)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcg.config import CONFIG
from bcg.datagen import trajectories as T
from bcg.datagen.sessions import make_session
from bcg.features.mouse import mouse_features, resample_sequence
from bcg.features.aggregate import session_to_features, sessions_to_matrix
from bcg.models.risk import score as risk_score


def _rng():
    return np.random.default_rng(0)


def test_generators_shapes():
    for fam in T.GENERATORS:
        tr = T.GENERATORS[fam](_rng(), 1280, 800)
        assert tr.ndim == 2 and tr.shape[1] == 3 and len(tr) >= 3


def test_human_more_curved_than_linear_bot():
    rng = _rng()
    human = mouse_features(T.gen_human(rng, 1280, 800))
    bot = mouse_features(T.gen_selenium_linear(rng, 1280, 800))
    # a linear bot is far straighter than a human path
    assert bot["straightness"] > human["straightness"]


def test_sequence_resample_fixed_length():
    tr = T.gen_human(_rng(), 1280, 800)
    seq = resample_sequence(tr, CONFIG.data.seq_len)
    assert seq.shape == (CONFIG.data.seq_len, 4)
    assert np.isfinite(seq).all()


def test_feature_vector_consistent_keys():
    rng = _rng()
    a = session_to_features(make_session(rng, "human", CONFIG))
    b = session_to_features(make_session(rng, "bezier_bot", CONFIG))
    assert set(a) == set(b) and len(a) > 30


def test_matrix_builds():
    rng = _rng()
    sessions = [make_session(rng, f, CONFIG) for f in T.GENERATORS] * 3
    X, y, names, fams = sessions_to_matrix(sessions)
    assert X.shape[0] == len(sessions) == len(y)
    assert np.isfinite(X).all()


def test_risk_blocks_obvious_bot():
    rng = _rng()
    bot = make_session(rng, "selenium_linear", CONFIG)
    r = risk_score(bot, model_p=0.9, cfg=CONFIG.risk)
    assert r.action in {"CHALLENGE", "BLOCK"}
    assert 0.0 <= r.risk <= 1.0


def test_risk_allows_confident_human():
    rng = _rng()
    human = make_session(rng, "human", CONFIG)
    human["session"]["webdriver_flag"] = False
    r = risk_score(human, model_p=0.05, cfg=CONFIG.risk)
    assert r.action == "ALLOW"
