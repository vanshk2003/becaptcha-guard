# Multimodal Deep-Learning CAPTCHA-Bypass Detection via Behavioural Analysis

> Draft sections for your report/paper. Replace bracketed placeholders
> (`[...]`), and confirm the numbers against your own training run
> (`artifacts/reports/metrics.json`). Figures referenced here are produced by
> `python -m bcg.train`, `python -m bcg.explain.shap_explain` and
> `python scripts/make_paper_figures.py`.

## Abstract (template)

We present **BeCAPTCHA-Guard**, a multimodal behavioural detector that
distinguishes genuine users from automated agents attempting to bypass CAPTCHA
challenges. The system captures mouse dynamics, keystroke timing, click patterns
and session-level interaction signals in the browser, and classifies sessions
with a deep 1-D CNN + BiLSTM sequence model backed by an adaptive, tiered
risk-scoring policy. Against four automation families — naive Selenium, stepwise
Puppeteer, "human-like" Bézier libraries, and an advanced minimum-jerk adversary
— the deep model attains [98.2]% accuracy and [0.998] ROC-AUC, and, crucially,
recovers [95.3]% recall on the human-like adversary where classical machine
learning collapses to [8–38]%. We contribute an end-to-end, explainable and
deployable pipeline with forensic logging.

## 1. Related work (positioning)

Our design draws on three lines of work. **BeCAPTCHA-Mouse** (Acien et al.,
*Pattern Recognition* 127:108643, 2022) established that synthetic mouse
trajectories generated from human motor-control models can train bot detectors;
we adopt and extend this synthesis methodology to four bot families and add
keystroke, click and session modalities. **BOTracle** (Kadel et al., ESORICS
2024 Workshops, LNCS 15264) motivates a layered pipeline of heuristics,
technical features and behavioural signals feeding a tiered risk decision; our
risk engine follows this defence-in-depth structure. **Breaking reCAPTCHAv2**
(Plesner et al., COMPSAC 2024) demonstrates the evasion behaviours — realistic
mouse movement and cookie/state reuse — that a modern detector must withstand;
these define our adversarial threat model, embodied by the *minimum-jerk
adversary* family.

## 2. Threat model

We consider an attacker who automates a real or headless browser to solve or
skip CAPTCHA flows. Capabilities range from (i) naive drivers that move the
cursor linearly and type with fixed delays, to (iv) an *adaptive adversary* that
reproduces human minimum-jerk motion, physiological-band tremor and irregular
event timing, and that hides environment tells (e.g., `navigator.webdriver`).
The detector must remain accurate against (iv) while keeping false positives on
genuine users low enough for a seamless experience.

## 3. Multimodal data-collection framework

A browser client instruments four modalities and posts them as one session
record to the backend, which stores each labelled trial in a SQLite datastore:

1. **Mouse dynamics** — `(x, y, t)` samples from `mousemove`, from which we
   derive velocity, acceleration, jerk, curvature, straightness and pauses.
2. **Keystroke dynamics** — per-key *dwell* time and inter-key *flight* time.
3. **Click patterns** — click count, inter-click intervals and button-press
   durations.
4. **Session metadata** — total duration, event rate, focus/scroll presence and
   the `navigator.webdriver` flag.

Collection uses a **guided two-task protocol** (Fig. 1 shows the resulting
trajectories): a *target-acquisition* task (acquire eight targets that appear at
random screen positions) elicits goal-directed mouse+click behaviour, and a
*typing* task (transcribe a fixed public phrase) elicits keystroke dynamics.
Each completed task is one labelled trial.

**Ethics and privacy.** Only interaction *timing and motion* are recorded,
keyed to a random anonymous participant ID; no personal identity and no free
keystroke content are stored (the typed text is a fixed public phrase). Explicit
informed consent is obtained before collection. For deployment with human
participants, obtain [institutional ethics / IRB] approval. [State your approval
reference here.]

### 3.1 Dataset

Genuine-user trials are collected from [N] participants via the web protocol.
Automation trials are gathered two ways: (a) by driving the same collection UI
with real Selenium/Puppeteer browsers (`scripts/bot_collect_selenium.py`),
yielding authentic automation telemetry; and (b) by synthesis from motor-control
models for controlled scale and reproducibility, following BeCAPTCHA-Mouse. The
released corpus contains [T] trials across five classes (human + four bot
families). Class-conditional feature distributions (Fig. 2) show why naive bots
separate trivially while the adversary overlaps the human distribution.

## 4. Feature extraction

Two views are computed per session. A **kinematic feature vector** (≈40
interpretable descriptors: straightness, velocity/acceleration/jerk statistics,
curvature, angle-change and pause statistics, clock regularity, plus keystroke,
click and session aggregates) feeds the classical baselines and SHAP
explainability. A **resampled trajectory sequence** — arc-length resampled to a
fixed length of 128 steps with channels `[dx, dy, dt, speed]` — feeds the deep
model, preserving the temporal *shape* of the movement.

A key modelling insight: real human movement peaks in speed *early*
(ballistic launch followed by feedback correction), producing a temporally
*asymmetric* velocity profile, whereas a minimum-jerk generator is symmetric.
This signal lives in the sequence shape, not in scalar aggregate statistics —
which is precisely why the sequence model outperforms feature-based classifiers
on the adversary.

## 5. Model

**BehaviorNet** stacks a two-layer 1-D CNN (local kinematic motifs) → BiLSTM
(long-range temporal structure) → additive attention pooling → a 2-class head.
The attention weights double as a sequence-level explanation of *where* in the
trajectory the decision was made. We compare against Logistic Regression, RBF
SVM, Random Forest and XGBoost on the kinematic feature vector.

## 6. Adaptive risk-scoring and response

Following BOTracle, decisions are layered. Layer 1 applies cheap
high-precision heuristics (webdriver flag, near-perfect straightness,
near-constant velocity, perfectly regular clock, too-few events). Layer 2 is the
learned `p(bot)`. Layer 3 fuses them into a single risk score mapped to a tiered
action: **ALLOW** (`risk < 0.35`), **CHALLENGE** (`0.35 ≤ risk < 0.75`) or
**BLOCK** (`risk ≥ 0.75`). Heuristics can only *raise* risk, protecting genuine
users from false blocks while catching obvious automation without model cost.

## 7. Experiments and results

Models are evaluated on a stratified held-out test split. We report accuracy,
precision, recall, F1, ROC-AUC and Equal Error Rate (EER), plus **per-family
recall** to test the adversary specifically.

**Table 1 — Overall detection (held-out test set).**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | EER |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.891 | 0.955 | 0.828 | 0.887 | 0.955 | 0.127 |
| RBF SVM | 0.861 | 0.952 | 0.770 | 0.851 | 0.929 | 0.163 |
| Random Forest | 0.881 | 0.985 | 0.781 | 0.871 | 0.939 | 0.131 |
| XGBoost | 0.893 | 0.943 | 0.844 | 0.891 | 0.957 | 0.115 |
| **BehaviorNet (ours)** | **0.982** | **0.977** | **0.988** | **0.983** | **0.998** | **0.022** |

**Table 2 — Per-family recall (fraction of each bot type caught).**

| Model | selenium | puppeteer | bezier | **minjerk_adv** | human TNR |
|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 1.000 | 0.984 | **0.328** | 0.958 |
| RBF SVM | 1.000 | 1.000 | 1.000 | **0.078** | 0.958 |
| Random Forest | 1.000 | 1.000 | 1.000 | **0.125** | 0.988 |
| XGBoost | 1.000 | 1.000 | 1.000 | **0.375** | 0.946 |
| **BehaviorNet (ours)** | 1.000 | 1.000 | 1.000 | **0.953** | 0.975 |

**Finding.** All models trivially catch naive automation, but classical
feature-based classifiers fail on the human-like adversary (7.8–37.5% recall).
BehaviorNet recovers 95.3% recall on this family by exploiting the temporal
velocity-asymmetry signature that aggregate features discard. This is the
central result motivating a sequence model for CAPTCHA-bypass detection.

## 8. Explainability

SHAP (TreeExplainer) attributions over the XGBoost baseline rank path
straightness, clock regularity, speed variability and curvature as the dominant
behavioural drivers of bot risk (Fig. 4) — physically consistent with the
kinematics of automation. For the deep model, attention weights localise the
decision along the trajectory and are surfaced per-request by the API.

## 9. Deployment, forensics and limitations

The detector is served as a real-time FastAPI endpoint (`/score`) with median
latency of a few milliseconds per session on CPU. Every decision is appended to
a hash-chained forensic log supporting tamper-evidence, satisfying audit
requirements. **Limitations:** results on the synthetic/authored corpus should
be confirmed on a larger field-collected human dataset; the adversary set,
though motivated by current evasion tooling, is not exhaustive; and adaptive
attackers may target the detector directly, motivating future adversarial
training and continual retraining.

## Figures

- **Fig. 1** `artifacts/reports/fig_trajectories.png` — representative
  trajectories by class.
- **Fig. 2** `artifacts/reports/fig_separability.png` — class-conditional
  feature distributions.
- **Fig. 3** `artifacts/reports/roc.png`, `confusion.png` — ROC curves and the
  deep model's confusion matrix.
- **Fig. 4** `artifacts/reports/shap_importance.png` — SHAP feature importance.

## References

1. A. Acien, A. Morales, J. Fierrez, R. Vera-Rodriguez. *BeCAPTCHA-Mouse:
   Synthetic mouse trajectories and improved bot detection.* Pattern Recognition
   127:108643, 2022.
2. [Kadel] et al. *BOTracle.* ESORICS 2024 Workshops, LNCS 15264.
3. [Plesner] et al. *Breaking reCAPTCHAv2.* COMPSAC 2024.
