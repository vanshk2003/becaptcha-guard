"""Synthetic mouse-trajectory generators.

The detector needs labelled human vs. bot motion. Real large-scale labelled
CAPTCHA telemetry is not publicly redistributable, so — following the
methodology of BeCAPTCHA-Mouse (Acien et al., Pattern Recognition 2022) — we
synthesise trajectories from models of human motor control and from the
kinematic signatures of common automation tools.

Each generator returns an (N, 3) float array of [x, y, t] samples (t in ms).

Human motion is built from the *minimum-jerk* model with additive sub-movements,
physiological tremor and target overshoot/correction — the signatures real
humans leave and that naive bots do not reproduce. Bot families model:
  * selenium_linear : linear interpolation / teleport-like constant velocity
  * puppeteer_step  : fixed-size grid steps, quantised timing
  * bezier_bot      : smooth cubic-Bezier "human-like" mouse libraries
  * minjerk_adv     : an ADVERSARY that also uses minimum-jerk + light tremor
                      (the hard case that a shallow detector misses)
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rand_endpoints(rng: np.random.Generator, w: int, h: int) -> tuple:
    start = np.array([rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.95) * h])
    end = np.array([rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.95) * h])
    # ensure a non-trivial distance
    while np.linalg.norm(end - start) < 0.15 * max(w, h):
        end = np.array([rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.95) * h])
    return start, end


def _minimum_jerk(t: Array) -> Array:
    """Normalised minimum-jerk position profile for t in [0, 1]."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _stack(x: Array, y: Array, t: Array) -> Array:
    return np.column_stack([x, y, t]).astype(np.float64)


# ---------------------------------------------------------------------------
# HUMAN
# ---------------------------------------------------------------------------
def gen_human(rng: np.random.Generator, w: int, h: int) -> Array:
    start, end = _rand_endpoints(rng, w, h)
    dist = np.linalg.norm(end - start)

    # Fitts'-law-ish movement time (ms), with heavy per-user variability
    mt = rng.uniform(0.55, 1.7) * (200 + 120 * np.log2(dist / 20 + 1))
    n = max(24, int(mt / 1000 * rng.uniform(55, 75)))  # ~60Hz-ish, jittered

    # non-uniform timestamps: humans do not sample on a perfect clock
    dt = rng.gamma(shape=6.0, scale=1.0, size=n)
    dt = dt / dt.sum() * mt
    t = np.cumsum(dt)
    tau = t / t[-1]

    # Human hallmark: velocity peaks EARLY (fast ballistic launch, then feedback
    # correction) -> a temporally *asymmetric* speed profile. We induce it with a
    # concave time-warp. This signal lives in the sequence SHAPE, not in scalar
    # mean/std features, which is exactly why a sequence model can exploit it.
    warp = rng.uniform(0.60, 0.90)
    prog = _minimum_jerk(tau ** warp)
    x = start[0] + (end[0] - start[0]) * prog
    y = start[1] + (end[1] - start[1]) * prog

    # 1-2 corrective sub-movements near the end (undershoot then correct)
    for _ in range(rng.integers(1, 3)):
        c = rng.uniform(0.55, 0.9)
        amp = rng.uniform(4, 22)
        ang = rng.uniform(0, 2 * np.pi)
        bump = amp * np.exp(-((tau - c) ** 2) / (2 * 0.03 ** 2))
        x += bump * np.cos(ang)
        y += bump * np.sin(ang)

    # physiological tremor (8-12 Hz) + fine motor noise
    freq = rng.uniform(8, 12)
    trem = rng.uniform(0.4, 1.4)
    x += trem * np.sin(2 * np.pi * freq * t / 1000 + rng.uniform(0, 6))
    y += trem * np.cos(2 * np.pi * freq * t / 1000 + rng.uniform(0, 6))
    x += rng.normal(0, 0.7, n)
    y += rng.normal(0, 0.7, n)

    # occasional mid-flight pause (distraction / re-plan)
    if rng.random() < 0.35:
        k = rng.integers(int(0.3 * n), int(0.8 * n))
        t[k:] += rng.uniform(60, 260)

    return _stack(x, y, t)


# ---------------------------------------------------------------------------
# BOT FAMILIES
# ---------------------------------------------------------------------------
def gen_selenium_linear(rng: np.random.Generator, w: int, h: int) -> Array:
    """Naive Selenium: essentially straight line, ~constant velocity, clock-steady."""
    start, end = _rand_endpoints(rng, w, h)
    n = int(rng.integers(15, 45))
    t = np.arange(n) * (1000.0 / rng.choice([30, 60, 120]))
    prog = np.linspace(0, 1, n)
    x = start[0] + (end[0] - start[0]) * prog
    y = start[1] + (end[1] - start[1]) * prog
    # tiny integer-pixel snapping noise only
    x += rng.normal(0, 0.15, n)
    y += rng.normal(0, 0.15, n)
    return _stack(x, y, t)


def gen_puppeteer_step(rng: np.random.Generator, w: int, h: int) -> Array:
    """Fixed-size steps toward target; quantised, monotone, no tremor."""
    start, end = _rand_endpoints(rng, w, h)
    step = rng.uniform(6, 16)
    pos = start.astype(float).copy()
    pts = [pos.copy()]
    for _ in range(400):
        d = end - pos
        dn = np.linalg.norm(d)
        if dn < step:
            break
        pos = pos + d / dn * step
        pts.append(pos.copy())
    pts.append(end.copy())
    pts = np.array(pts)
    n = len(pts)
    t = np.arange(n) * rng.choice([8, 10, 16])  # quantised timing
    return _stack(pts[:, 0], pts[:, 1], t.astype(float))


def _cubic_bezier(p0, p1, p2, p3, u):
    return (
        (1 - u) ** 3 * p0[:, None]
        + 3 * (1 - u) ** 2 * u * p1[:, None]
        + 3 * (1 - u) * u**2 * p2[:, None]
        + u**3 * p3[:, None]
    )


def gen_bezier_bot(rng: np.random.Generator, w: int, h: int) -> Array:
    """'Human-like' cubic-Bezier libraries: smooth, but symmetric velocity,
    no tremor, and evenly-clocked samples."""
    start, end = _rand_endpoints(rng, w, h)
    mid = (start + end) / 2
    off = rng.uniform(-1, 1, 2) * np.linalg.norm(end - start) * 0.25
    c1 = mid + off + rng.uniform(-30, 30, 2)
    c2 = mid - off + rng.uniform(-30, 30, 2)
    n = int(rng.integers(30, 70))
    u = np.linspace(0, 1, n)
    pts = _cubic_bezier(start, c1, c2, end, u)
    t = np.arange(n) * (1000.0 / rng.choice([50, 60, 60, 100]))
    x = pts[0] + rng.normal(0, 0.25, n)
    y = pts[1] + rng.normal(0, 0.25, n)
    return _stack(x, y, t)


def gen_minjerk_adv(rng: np.random.Generator, w: int, h: int) -> Array:
    """Advanced adversary (the HARD case, à la 'Breaking reCAPTCHAv2').

    A near-perfect mimic: gamma-jittered clock, human-band tremor, occasional
    corrective sub-movements and distraction pauses. Its residual tells are
    subtle and mostly *temporal*:
      * its speed profile is (near-)symmetric in time (min-jerk peaks at 50%),
        whereas real humans peak early;
      * sub-movements are slightly smaller and rarer.
    We deliberately let its parameters OVERLAP the human range so the two
    classes are not linearly separable — this is what makes the benchmark
    honest and what the sequence model must resolve.
    """
    start, end = _rand_endpoints(rng, w, h)
    dist = np.linalg.norm(end - start)
    mt = rng.uniform(0.6, 1.6) * (200 + 120 * np.log2(dist / 20 + 1))
    n = max(24, int(mt / 1000 * rng.uniform(55, 75)))

    dt = rng.gamma(shape=6.0, scale=1.0, size=n)      # same clock model as humans
    dt = dt / dt.sum() * mt
    t = np.cumsum(dt)
    tau = t / t[-1]

    # near-symmetric warp (the tell): centred at ~1.0 with slight overlap into
    # the human range, so some adversary samples are genuinely ambiguous.
    warp = rng.uniform(0.92, 1.06)
    prog = _minimum_jerk(np.clip(tau ** warp, 0, 1))
    x = start[0] + (end[0] - start[0]) * prog
    y = start[1] + (end[1] - start[1]) * prog

    # fewer / smaller corrective sub-movements than a human
    for _ in range(rng.integers(0, 2)):
        c = rng.uniform(0.55, 0.9)
        amp = rng.uniform(2, 12)
        ang = rng.uniform(0, 2 * np.pi)
        bump = amp * np.exp(-((tau - c) ** 2) / (2 * 0.03 ** 2))
        x += bump * np.cos(ang)
        y += bump * np.sin(ang)

    freq = rng.uniform(8, 12)                          # human-band tremor
    trem = rng.uniform(0.3, 1.1)
    x += trem * np.sin(2 * np.pi * freq * t / 1000 + rng.uniform(0, 6))
    y += trem * np.cos(2 * np.pi * freq * t / 1000 + rng.uniform(0, 6))
    x += rng.normal(0, 0.7, n)
    y += rng.normal(0, 0.7, n)

    if rng.random() < 0.2:                             # occasional pause
        k = rng.integers(int(0.3 * n), int(0.8 * n))
        t[k:] += rng.uniform(60, 220)
    return _stack(x, y, t)


GENERATORS = {
    "human": gen_human,
    "selenium_linear": gen_selenium_linear,
    "puppeteer_step": gen_puppeteer_step,
    "bezier_bot": gen_bezier_bot,
    "minjerk_adv": gen_minjerk_adv,
}
