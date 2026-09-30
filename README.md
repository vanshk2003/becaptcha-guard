# BeCAPTCHA-Guard

**Multimodal deep-learning detection of CAPTCHA-bypass bots via behavioural analysis.**

A defensive security system that tells genuine users apart from automated agents
(Selenium, Puppeteer, "human-like" bots) by analysing how they *behave* — mouse
dynamics, keystroke timing, click patterns and session signals — rather than by
posing puzzles. It ships an end-to-end pipeline: multimodal data collection, a
deep sequence model with classical baselines, an adaptive tiered risk engine,
explainability, forensic logging, and a real-time API with a browser UI.

> Built around three works: **BeCAPTCHA-Mouse** (Acien et al., Pattern
> Recognition 2022) for behavioural synthesis, **BOTracle** (ESORICS 2024) for
> the layered risk pipeline, and **Breaking reCAPTCHAv2** (Plesner et al.,
> COMPSAC 2024) as the adversarial threat model.

## Headline result

Every model catches naive automation; only the deep model catches the
*human-like* adversary.

| Model | Accuracy | ROC-AUC | EER | Recall on human-like adversary |
|---|---|---|---|---|
| XGBoost (best classical) | 0.893 | 0.957 | 0.115 | **0.375** |
| **BehaviorNet (deep, ours)** | **0.982** | **0.998** | **0.022** | **0.953** |

## Architecture

```
 Browser collector ──POST──▶ FastAPI ──▶ feature extraction ──▶ BehaviorNet (1D-CNN + BiLSTM + attention)
 (mouse/keys/clicks)           │                └─▶ classical baselines (RF/SVM/XGB/LogReg)
        │                      ▼
 SQLite datastore        adaptive risk engine (heuristics + model → ALLOW/CHALLENGE/BLOCK)
        │                      │
 dataset explorer         hash-chained forensic log
```

## Quickstart

```bash
git clone <your-repo-url> becaptcha-guard && cd becaptcha-guard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) train models + produce metrics/plots (writes to artifacts/)
PYTHONPATH=src python -m bcg.train
PYTHONPATH=src python -m bcg.explain.shap_explain
python scripts/make_paper_figures.py

# 2) run the service (collector, explorer, API)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open:

| URL | What |
|---|---|
| `http://localhost:8000/` | **Data collection** UI (consent + guided tasks) |
| `http://localhost:8000/explore` | **Dataset explorer** (stats, distributions, trajectory gallery, export) |
| `http://localhost:8000/demo` | Live single-session **risk demo** |
| `http://localhost:8000/dashboard` | **Forensic** dashboard (live decisions, chain integrity) |
| `http://localhost:8000/docs` | Interactive API docs |

## Collecting data

- **Humans:** open `/` , give consent, complete the two tasks. Each task stores a
  labelled trial.
- **Bots (synthetic, fast):** with the API running,
  `python scripts/seed_dataset.py --url http://localhost:8000 --per-family 60 --humans 0`
- **Bots (authentic, via a real browser):**
  `pip install selenium && python scripts/bot_collect_selenium.py --label selenium_linear --trials 20`
- **Export:** `GET /dataset/export.csv` (analysis-ready features) or
  `GET /dataset/export.jsonl` (raw sessions). Both are buttons in `/explore`.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + loaded models |
| `POST` | `/score` | score a session → risk + tiered action + explanation |
| `POST` | `/collect` | persist one labelled trial |
| `GET` | `/dataset/stats` | corpus statistics |
| `GET` | `/dataset/sessions` | list trials |
| `GET` | `/dataset/session/{id}` | one raw session |
| `GET` | `/dataset/export.{csv,jsonl}` | export corpus |
| `GET` | `/events` | forensic log + hash-chain verification |

## Repository layout

```
src/bcg/            core library
  datagen/          synthetic human + bot generators, session assembly
  features/         mouse/keystroke/click/session features + sequence resampling
  models/           deep.py (BehaviorNet), baselines.py, risk.py (tiered policy)
  explain/          SHAP explainability
  train.py          end-to-end train + evaluate + save artifacts
  store.py          SQLite datastore
api/                FastAPI service (main.py, schemas.py)
web/                collect.html, explore.html, collector.html, dashboard.html
scripts/            seed_dataset.py, bot_collect_selenium.py, make_paper_figures.py
paper/              methodology.md  (paper-ready sections + results tables)
tests/              pipeline tests
artifacts/          models, reports, dataset (generated; git-ignored)
```

## Docker

```bash
docker compose up --build      # trains on first boot, then serves on :8000
```

## Testing

```bash
PYTHONPATH=. pytest -q
```

## Notes and limitations

The included corpus is synthetic/authored for reproducibility; confirm results on
a larger field-collected human dataset before drawing strong conclusions. The
adversary set is representative, not exhaustive. This is a research/educational
project — a production deployment needs field data, adversarial hardening,
rate-limiting and institutional ethics approval for human-subject collection.

## License

MIT — see `LICENSE`.
