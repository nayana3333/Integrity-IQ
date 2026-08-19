<div align="center">

# 🛡️ IntegrityIQ

**An adaptive, multi-signal academic-integrity platform — not another string-matching plagiarism scanner.**

[![CI](https://github.com/nayana3333/IntegrityIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/nayana3333/IntegrityIQ/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/agents-LangGraph-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![IBM Granite](https://img.shields.io/badge/LLM-IBM%20Granite-052FAD?logo=ibm&logoColor=white)](https://www.ibm.com/granite)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Overview](#overview) · [Demo](#demo) · [How it works](#how-it-works) · [Evaluation](#evaluation) · [Quick start](#quick-start) · [Architecture](#architecture--design-decisions)

</div>

---

## Overview

Most plagiarism tools do one thing: string-match a submission against a web index. That catches copy-paste, and nothing else. It misses **paraphrased plagiarism**, **AI-generated submissions**, and **contract cheating/ghostwriting** (where the text is original and matches nothing online — so there is nothing to string-match against).

IntegrityIQ fuses three independent, complementary detection signals, calibrates their combined weight **per course from real instructor feedback**, and has an **IBM Granite**-powered agent turn the result into an evidence-cited report instead of a bare score — because a false accusation is a worse failure mode here than a missed detection, and instructors need to see *why* something was flagged, not just *that* it was.

Built as the capstone project for the **IBM SkillsBuild × AICTE × Edunet Foundation** Agentic AI internship (Problem Statement #10 — *AI-Driven Plagiarism Intelligence for Assignments*), and taken well past the minimum bar: every detection signal is independently evaluated against a public benchmark with real, honestly-reported numbers (see [Evaluation](#evaluation)) — including where a signal is weak, and why.

## Demo

> 📸 **Screenshots and a walkthrough video go here.** The app is fully working locally (see [Quick start](#quick-start)) — screenshots of the login screen, an upload + risk report, the flagged-submissions list, and the style-drift chart belong in `docs/screenshots/`, referenced below. A 60–90s screen recording of the golden path (upload → risk score → flagged evidence → confirm/reject → retrain) is the single most persuasive thing in this README; add it above the fold once captured.

<!--
Once captured, replace this block with something like:

![Login screen](docs/screenshots/01-login.png)
![Upload & risk report](docs/screenshots/02-upload-report.png)
![Flagged submissions](docs/screenshots/03-flagged.png)
![Style-drift trend](docs/screenshots/04-style-trend.png)

https://github.com/user-attachments/assets/<demo-video-id>
-->

## How it works

| Signal | Catches | How |
|---|---|---|
| **Semantic similarity** | Paraphrased plagiarism between classmates | Chunk-level sentence-transformer embeddings, brute-force cosine search over a persistent NumPy index of every past submission in the course |
| **AI-generated-text likelihood** | ChatGPT/LLM-written submissions | Perplexity + burstiness features under a reference LM, fed into a classifier trained on the HC3 human-vs-ChatGPT dataset |
| **Stylometric drift** | Ghostwriting / contract cheating (matches nothing online, so the above two miss it) | Per-student writing-style fingerprint (function-word frequencies, sentence rhythm, punctuation habits) built with an online Welford mean/variance update; new submissions are scored against the student's *own* baseline |

These are fused into one calibrated risk score by a per-course `RiskFusionModel` — a logistic regression that starts from a conservative, documented default and **retrains on accumulated instructor verdicts** (confirmed / false-positive) once a course has enough labeled feedback. That adaptive retraining loop is what turns this from a static scanner into the "learns from historical submissions and instructor feedback" system the problem statement calls for.

A LangGraph-orchestrated agent then takes the fused score plus the raw evidence and asks **IBM Granite** to write an instructor-facing report — instructed to cite specific matched passages and hedge on low-confidence signals, never to assert misconduct outright.

## ✨ Features

- 🔍 **Three independent detectors, not one** — semantic similarity, AI-text classification, and stylometric drift, each individually evaluated (not just glued together and hoped for)
- 🧠 **Adaptive per-course model** — retrains its signal weighting from instructor confirm/false-positive feedback; two courses with different assignment styles end up differently calibrated
- 🤖 **Agentic, evidence-based reporting** — a LangGraph `detect → explain` graph, with IBM Granite writing hedged, citation-backed reports instead of a bare "97% risk" number
- 👻 **Catches ghostwriting** — the one form of misconduct that similarity search and AI-detectors structurally cannot see, via a per-student writing-style fingerprint
- 📊 **Real, honest evaluation** — every signal benchmarked against a public dataset (HC3, PAWS, an authorship-attribution corpus), with weaknesses documented, not hidden
- 🖥️ **Full instructor console** — upload, review flagged passages side-by-side, confirm/reject, watch style-drift trends over a semester, trigger retraining — all in a polished Streamlit dashboard
- 🐳 **Deployable** — Dockerized, with an IBM Cloud Code Engine deployment guide

## Evaluation

Unlike a hand-wavy "it seems to work" plagiarism tool, each signal is evaluated independently against a public benchmark before being trusted in the fused score. Full numbers in [`eval/results/*.json`](eval/results); summary:

| Signal | Dataset | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **AI-text detector** | [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (700 human vs. ChatGPT answers, held-out test split) | 0.91 | 0.99 | **0.95** | **0.99** |
| **Semantic similarity** — realistic task | [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) paraphrases vs. randomly-paired unrelated text (488 pairs) | 0.99 | 0.98 | **0.99** | **1.00** |
| Semantic similarity — adversarial task | PAWS's own hard-negative labels (600 pairs, same words/different meaning) | 0.41 | 0.98 | 0.58 | 0.63 |
| **Stylometric drift** | [ELI5 authorship-attribution](https://huggingface.co/datasets/manu/eli5_authorship_attribution), 150 authors, same- vs. cross-author (~750-word pseudo-documents) | 0.56 | 0.91 | 0.69 | 0.65 |

Two results are worth explaining rather than glossing over:

- **Similarity: two very different numbers for the same model, on purpose.** PAWS's own task is deliberately adversarial — "the dog bit the man" vs. "the man bit the dog" share every word but mean the opposite thing, and a generic (not paraphrase-fine-tuned) sentence embedding is *expected* to struggle there (F1 0.58 — a known, documented limitation, not a bug). The "realistic" row re-evaluates the same model on the case that actually matters here — genuine paraphrase vs. ordinary unrelated text, i.e. what a copied-and-reworded paragraph vs. a different student's different essay actually looks like — and it's excellent (F1 0.99). Reporting both, rather than only the flattering one, is the point: it shows *where* the signal is strong and *where* it isn't.
- **Stylometric drift is the weakest of the three signals (F1 0.69, ROC-AUC 0.65 vs. 0.95–1.00 for the other two) — and that's reflected in the design, not hidden from it.** `RiskFusionModel`'s default weights give it the lowest coefficient (see `app/fusion/risk_model.py`), because style is inherently noisier than semantic content: it needs long samples (500+ words — classic authorship-attribution literature bears this out) and several prior submissions before it's reliable at all. It's kept in the system anyway because it's the *only* signal that can catch ghostwriting — text that matches nothing online and isn't AI-generated, so the other two are structurally blind to it.

Reproduce with:
```bash
python -m eval.scripts.train_ai_detector
python -m eval.scripts.eval_similarity_detector
python -m eval.scripts.eval_stylometry_detector
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + SQLAlchemy | async-friendly, typed, auto-generated OpenAPI docs |
| Dashboard | Streamlit + a custom CSS design system (`dashboard/theme.py`) | fast to build, still looks like a real product instead of a stock widget demo |
| Agent orchestration | LangGraph | explicit, inspectable `detect → explain` state graph rather than a hidden prompt-chaining loop |
| LLM | IBM Granite (watsonx.ai, or a local checkpoint via `transformers`) | swappable backend behind one interface — runs on the mandated IBM Cloud Lite path *and* fully offline for CI/local dev |
| Embeddings | `sentence-transformers` (MiniLM) + a brute-force NumPy cosine index | see [Architecture](#architecture--design-decisions) for why not a vector DB |
| AI-text detection | GPT-2 perplexity/burstiness features → `scikit-learn` logistic regression | trained and evaluated on HC3, not hand-tuned thresholds |
| Auth | JWT (`python-jose`) + `passlib`/`bcrypt` | standard, stateless |
| Storage | SQLite (dev) / swappable via `DATABASE_URL` | zero-setup local dev, Postgres-ready for production |
| Testing | `pytest` | 9 tests covering ingestion, stylometry, fusion, embeddings, AI-detection, and full agent-orchestration integration |
| Linting | `ruff` | zero warnings across the whole tree, enforced in CI |
| Deployment | Docker + Docker Compose, IBM Cloud Code Engine | see `deploy/DEPLOY.md` |

## Architecture & design decisions

```
┌─────────────┐      ┌──────────────────────────────────────────────┐
│  Streamlit   │      │                  FastAPI                     │
│  Dashboard   │◄────►│  routes: auth / courses / submissions /      │
│ (instructor) │      │          feedback                             │
└─────────────┘      └──────────────────────┬───────────────────────┘
                                             │
                              LangGraph orchestration (app.agents)
                              ┌──────────────┴──────────────┐
                              │  detect ──────────► explain  │
                              └──────┬───────────────┬───────┘
                                     │               │
                     ┌───────────────┴───┐   ┌───────┴────────┐
                     │ DetectionPipeline  │   │ IBM Granite    │
                     │ (app.pipeline)     │   │ (watsonx.ai /  │
                     └──┬──────┬──────┬──┘   │  local)        │
                        │      │      │      └────────────────┘
              ┌─────────┘  ┌───┘   ┌──┘
       similarity      ai-text   style
       (NumPy +        detector  drift
       MiniLM)         (GPT-2    (Welford
                        perplexity fingerprint)
                        + LR clf)
                        │
                 fused by RiskFusionModel (per-course, retrains on
                 instructor feedback via FeedbackLoopAgent)
```

Design choices worth calling out (all deliberate, not defaults):

- **The detection core (`app/pipeline.py`, `app/stylometry`, `app/ai_detector`, `app/embeddings`, `app/fusion`) has zero FastAPI/SQLAlchemy imports.** It takes plain callbacks for persistence. That's what lets the exact same code be unit-tested, run from the offline eval harness, and called from the agent layer without spinning up a server or database.
- **Diagonal, not full, covariance for the Mahalanobis-style style-drift score.** A full covariance matrix needs far more samples than a course ever produces per student (70+ stylometric features vs. a handful of essays) — the diagonal approximation is the standard, explicitly-chosen tradeoff for small-sample stylometry.
- **Brute-force NumPy cosine search, not a vector database, for semantic similarity.** ChromaDB was the first choice, but its compiled Rust extension crashed (a native-dependency reproducibility issue, not a bug in this codebase) during testing. Rather than chase that, the index was rewritten as a plain `(N, 384)` NumPy matrix with a single matmul per query — which is also just the *correct* choice at course scale (a few thousand chunks), since an ANN index only starts paying for its complexity past ~100K+ vectors. Swappable behind the same `VectorStore` interface if a deployment ever needs to scale past that.
- **The explanation agent is instructed to under-claim, not over-claim** — it cites evidence and hedges, never asserts misconduct. A wrongly confident false accusation is worse than a missed detection.
- **Granite backend is swappable** (`GRANITE_BACKEND=watsonx` vs. `local`) so the project runs both on the mandated IBM Cloud Lite / watsonx.ai path and fully offline for CI/local dev.

## Quick start

```bash
git clone https://github.com/nayana3333/IntegrityIQ.git
cd IntegrityIQ

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # fill in watsonx credentials, or leave GRANITE_BACKEND=local
```

Run the API and dashboard in two terminals:

```bash
uvicorn app.api.main:app --reload      # terminal 1 — API on :8000
streamlit run dashboard/app.py         # terminal 2 — dashboard on :8501
```

Open **http://localhost:8501**, register an instructor account, create a course, add a student, and upload a submission — you'll get a risk score, severity, and an evidence-cited report in a few seconds.

Or with Docker:
```bash
docker compose up --build
```

See [`deploy/DEPLOY.md`](deploy/DEPLOY.md) for IBM Cloud Lite / watsonx.ai credential setup and IBM Cloud Code Engine deployment.

### Running the tests

```bash
pytest tests/ -v        # 9 tests: ingestion, stylometry, fusion, embeddings, AI-detection, full agent integration
ruff check .            # zero warnings, enforced in CI
```

## Project structure

```
app/
  ingestion/    PDF/DOCX/TXT parsing, sentence-aware chunking
  embeddings/   sentence-transformer + NumPy brute-force cosine-similarity index
  stylometry/   per-student style fingerprint + drift scoring
  ai_detector/  perplexity/burstiness features + trained classifier
  fusion/       per-course risk-fusion model, retrainable from feedback
  agents/       LangGraph orchestration + Granite client + feedback loop
  api/          FastAPI app (auth, courses, submissions, feedback)
  db/           SQLAlchemy models
  pipeline.py   wires the four detection modules together
dashboard/      Streamlit instructor console + custom design system (theme.py)
eval/           evaluation scripts + results (real, reproducible numbers)
deploy/         Dockerfiles + IBM Cloud deployment guide
docs/           resume/interview prep, screenshots
tests/          unit + integration tests for the detection core and agent layer
```

## API reference

Full interactive docs (Swagger UI) are auto-generated at **`/docs`** when the API is running. Key endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/register`, `/auth/login` | instructor account + JWT auth |
| `POST` | `/courses` | create a course |
| `POST` | `/courses/{id}/students` | add a student |
| `POST` | `/courses/{id}/students/{id}/submissions` | upload + run the full detect→explain pipeline |
| `GET` | `/courses/{id}/submissions` | list all analyzed submissions, sorted by risk |
| `POST` | `/submissions/{id}/feedback` | instructor confirms/rejects a flag (feeds the adaptive loop) |
| `POST` | `/courses/{id}/retrain` | retrain that course's fusion model on accumulated feedback |

## Roadmap / honest limitations

- The AI-text detector is trained on HC3, which skews toward Q&A-style text; a course with a very different genre (creative writing, code comments) would benefit from fine-tuning on in-domain examples.
- Style-drift confidence is low until a student has 3+ prior submissions on file — by design, but it does mean the first assignment of a semester gets weaker signal.
- The fusion model currently caches per-course in-process memory; a multi-worker production deployment would move that to Redis or reload-from-disk per request.
- No web-search fallback yet (checking against sources *outside* the course, not just other students' submissions) — would slot in as an additional LangGraph tool node without changing the fusion/agent architecture.

## Contributing

This started as a solo capstone project, but issues and PRs are welcome — particularly around the roadmap items above. Please run `pytest` and `ruff check .` before opening a PR; CI enforces both.

## License

[MIT](LICENSE) © 2026 Nayana S

## Credits

Built by **Nayana S** as the capstone project for the **IBM SkillsBuild for University Engagements — AICTE 2026** Agentic AI internship, in collaboration with Edunet Foundation.
