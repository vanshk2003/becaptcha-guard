"""Synthetic keystroke-dynamics generators.

Keystroke timing is a classic behavioural biometric. We model two observables
per keypress:
  * dwell time  (key-down -> key-up of the same key)
  * flight time (key-up of key i -> key-down of key i+1)

Humans show log-normal, digraph-dependent, positively-correlated timings with
occasional long pauses. Automation either types with fixed delays (paste/
sendKeys) or with unnaturally low variance.

Returns a dict with 'dwell' and 'flight' arrays (ms).
"""
from __future__ import annotations

import numpy as np


def gen_human_keystrokes(rng: np.random.Generator, n_chars: int | None = None) -> dict:
    n = n_chars or int(rng.integers(8, 22))
    # log-normal dwell, per-user speed
    mu = rng.uniform(np.log(70), np.log(110))
    dwell = rng.lognormal(mu, 0.28, n)
    flight = rng.lognormal(rng.uniform(np.log(90), np.log(170)), 0.42, max(1, n - 1))
    # sprinkle in a few "thinking" pauses
    for _ in range(rng.integers(0, 3)):
        if len(flight):
            flight[rng.integers(0, len(flight))] += rng.uniform(150, 600)
    return {"dwell": dwell, "flight": flight}


def gen_bot_keystrokes(rng: np.random.Generator, kind: str, n_chars: int | None = None) -> dict:
    n = n_chars or int(rng.integers(8, 22))
    if kind == "selenium_linear":
        # sendKeys: nearly constant, tiny variance
        dwell = np.full(n, rng.uniform(30, 60)) + rng.normal(0, 1.5, n)
        flight = np.full(max(1, n - 1), rng.uniform(5, 25)) + rng.normal(0, 1.0, max(1, n - 1))
    elif kind == "puppeteer_step":
        # type() with fixed delay option
        d = rng.choice([40, 50, 80, 100])
        dwell = np.full(n, d) + rng.normal(0, 3, n)
        flight = np.full(max(1, n - 1), d) + rng.normal(0, 3, max(1, n - 1))
    else:  # bezier_bot / minjerk_adv: try to look human but under-dispersed
        dwell = rng.lognormal(np.log(85), 0.12, n)
        flight = rng.lognormal(np.log(120), 0.16, max(1, n - 1))
    dwell = np.clip(dwell, 5, None)
    flight = np.clip(flight, 1, None)
    return {"dwell": dwell, "flight": flight}
