"""BeCAPTCHA-Guard real-time detection API.

Endpoints
  GET  /health           liveness + which models are loaded
  POST /score            score a multimodal session -> risk + tiered action
  GET  /                 serves the browser telemetry collector (demo)
  GET  /dashboard        serves the forensic dashboard

Every decision is appended to a tamper-evident forensic log
(artifacts/forensic/events.jsonl) with a hash chain, satisfying the
'forensic logging' objective. Real deployments would ship this to a SIEM.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from bcg.infer import get_detector          # noqa: E402
from bcg.config import ARTIFACTS            # noqa: E402
from bcg import store                        # noqa: E402
from api.schemas import (SessionIn, ScoreOut,  # noqa: E402
                         CollectIn, CollectOut)
from fastapi.responses import PlainTextResponse  # noqa: E402
from fastapi import Response                  # noqa: E402

WEB = ROOT / "web"
FORENSIC = ARTIFACTS / "forensic"
FORENSIC.mkdir(parents=True, exist_ok=True)
LOG = FORENSIC / "events.jsonl"

app = FastAPI(title="BeCAPTCHA-Guard", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _prev_hash() -> str:
    if not LOG.exists():
        return "0" * 64
    last = None
    with LOG.open() as f:
        for line in f:
            last = line
    if not last:
        return "0" * 64
    try:
        return json.loads(last)["hash"]
    except Exception:
        return "0" * 64


def _forensic_write(record: dict) -> str:
    prev = _prev_hash()
    payload = json.dumps(record, sort_keys=True)
    h = hashlib.sha256((prev + payload).encode()).hexdigest()
    entry = {**record, "prev_hash": prev, "hash": h}
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return h


@app.get("/health")
def health():
    det = get_detector()
    return {"status": "ok", "deep_model": "BehaviorNet",
            "baselines": list(det.baselines.keys()), "seq_len": det.seq_len}


@app.post("/score", response_model=ScoreOut)
def score(inp: SessionIn):
    det = get_detector()
    session = inp.model_dump()
    baseline = session.pop("baseline", "xgboost")
    t0 = time.time()
    result = det.score(session, baseline=baseline)
    latency_ms = round((time.time() - t0) * 1000, 2)
    rid = str(uuid.uuid4())

    _forensic_write({
        "request_id": rid,
        "ts": time.time(),
        "action": result["action"],
        "risk": result["risk"],
        "p_bot_deep": result["p_bot_deep"],
        "p_bot_baseline": result["p_bot_baseline"],
        "reasons": result["reasons"],
        "latency_ms": latency_ms,
        "n_mouse_points": len(session["mouse"]["x"]),
    })

    out = {k: result[k] for k in
           ("risk", "action", "p_bot_deep", "p_bot_baseline",
            "baseline_model", "reasons", "attention_peaks_path_fraction")}
    out["request_id"] = rid
    return JSONResponse(out)


@app.get("/events")
def events(limit: int = 200):
    if not LOG.exists():
        return {"events": [], "chain_valid": True}
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    # verify hash chain (forensic integrity check)
    prev = "0" * 64
    valid = True
    for r in rows:
        payload = {k: r[k] for k in r if k not in ("prev_hash", "hash")}
        h = hashlib.sha256((prev + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
        if h != r.get("hash") or r.get("prev_hash") != prev:
            valid = False
            break
        prev = r["hash"]
    return {"events": rows[-limit:], "chain_valid": valid, "total": len(rows)}


# --------------------------------------------------------------------------
# Data collection + dataset management
# --------------------------------------------------------------------------
@app.post("/collect", response_model=CollectOut)
def collect(inp: CollectIn):
    """Persist one labelled trial to the datastore (this is the corpus builder)."""
    data = inp.model_dump()
    meta = {k: data.pop(k) for k in
            ("participant", "label", "task", "consent", "user_agent")}
    sid = store.insert_session(data, meta)
    return {"id": sid, "stored": True, "label": meta["label"]}


@app.get("/dataset/stats")
def dataset_stats():
    return store.stats()


@app.get("/dataset/sessions")
def dataset_sessions(limit: int = 60, label: str | None = None):
    return {"sessions": store.list_sessions(limit=limit, label=label)}


@app.get("/dataset/session/{sid}")
def dataset_session(sid: str):
    s = store.get_session(sid)
    return s or JSONResponse({"error": "not found"}, status_code=404)


@app.get("/dataset/export.jsonl")
def export_jsonl():
    return Response(store.export_jsonl(), media_type="application/x-ndjson",
                    headers={"Content-Disposition": "attachment; filename=becaptcha_dataset.jsonl"})


@app.get("/dataset/export.csv")
def export_csv():
    return Response(store.export_features_csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=becaptcha_features.csv"})


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.get("/")
def home():
    return FileResponse(WEB / "collect.html")


@app.get("/collect-ui")
def collect_ui():
    return FileResponse(WEB / "collect.html")


@app.get("/explore")
def explore():
    return FileResponse(WEB / "explore.html")


@app.get("/demo")
def collector():
    return FileResponse(WEB / "collector.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse(WEB / "dashboard.html")
