"""Populate the datastore by posting generated labelled sessions to /collect.

Useful for (a) smoke-testing the running API, (b) bulk-generating bot-family
trials at scale, and (c) giving the dataset explorer content for screenshots.
Genuine HUMAN sessions should still be collected from real participants through
the web UI (/collect-ui); this script is for the programmatic bot families and
for demos.

Run (API must be running):
    python scripts/seed_dataset.py --url http://127.0.0.1:8000 --per-family 60 --humans 120
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import urllib.request
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bcg.datagen.sessions import make_session
from bcg.config import CONFIG


def post(url, payload):
    req = urllib.request.Request(url + "/collect", method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps(payload).encode())
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--humans", type=int, default=120)
    ap.add_argument("--per-family", type=int, default=60)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    families = ["selenium_linear", "puppeteer_step", "bezier_bot", "minjerk_adv"]
    plan = [("human", args.humans)] + [(f, args.per_family) for f in families]

    total = 0
    for fam, n in plan:
        for i in range(n):
            s = make_session(rng, fam, CONFIG)
            payload = {
                "mouse": s["mouse"], "keystroke": s["keystroke"],
                "clicks": s["clicks"], "session": s["session"],
                "participant": f"SEED-{fam}-{i % 12:02d}", "label": fam,
                "task": "target_acquisition" if i % 2 else "typing",
                "consent": True, "user_agent": "seed-script",
            }
            post(args.url, payload)
            total += 1
        print(f"  seeded {n:>4} × {fam}")
    print(f"done: {total} sessions posted to {args.url}")


if __name__ == "__main__":
    main()
