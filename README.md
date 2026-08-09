# IntegrityIQ

**An adaptive, multi-signal academic-integrity assistant.** Built as the
capstone project for the IBM SkillsBuild × AICTE × Edunet Foundation Agentic
AI internship (Problem Statement #10 — *AI-Driven Plagiarism Intelligence for
Assignments*).

Most plagiarism tools do one thing: string-match a submission against a web
index. That catches copy-paste, and nothing else. It misses paraphrased
plagiarism, AI-generated submissions, and contract cheating/ghostwriting
(where the text is original and matches nothing online). IntegrityIQ instead
fuses three independent, complementary signals, calibrates their combined
weight **per course from real instructor feedback**, and has an IBM
Granite-powered agent turn the result into an evidence-cited report instead
of a bare score — because a false accusation is a worse failure mode here
than a missed detection, and instructors need to be able to see *why*
something was flagged, not just *that* it was.

## The three signals

| Signal | Catches | How |
|---|---|---|
| **Semantic similarity** | Paraphrased plagiarism between classmates | Chunk-level sentence-transformer embeddings, brute-force cosine search over a persistent NumPy index of every past submission in the course |
| **AI-generated-text likelihood** | ChatGPT/LLM-written submissions | Perplexity + burstiness features under a reference LM, fed into a classifier trained on the HC3 human-vs-ChatGPT dataset |
| **Stylometric drift** | Ghostwriting / contract cheating (matches nothing online, so the above two miss it) | Per-student writing-style fingerprint (function-word frequencies, sentence rhythm, punctuation habits) built with an online Welford mean/variance update; new submissions are scored against the student's *own* baseline |

These are fused into one calibrated risk score by a per-course
`RiskFusionModel` — a logistic regression that starts from a conservative,
documented default and **retrains on accumulated instructor verdicts**
(confirmed / false-positive) once a course has enough labeled feedback. That
adaptive retraining loop is what turns this from a static scanner into the
"learns from historical submissions and instructor feedback" system the
problem statement asks for.

## Architecture

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

- **The detection core (`app/pipeline.py`, `app/stylometry`, `app/ai_detector`,
  `app/embeddings`, `app/fusion`) has zero FastAPI/SQLAlchemy imports.** It
  takes plain callbacks for persistence. That's what lets the exact same code
  be unit-tested, run from the offline eval harness, and called from the
  agent layer without spinning up a server or database.
- **Diagonal, not full, covariance for the Mahalanobis-style style-drift
  score.** A full covariance matrix needs far more samples than a course
  ever produces per student (70+ stylometric features vs. a handful of
  essays) — the diagonal approximation is the standard, explicitly-chosen
  tradeoff for small-sample stylometry.
- **Brute-force NumPy cosine search, not a vector database, for semantic
  similarity.** ChromaDB was the first choice, but its compiled Rust
  extension crashed (a native-dependency reproducibility issue, not a bug in
  this codebase) during testing. Rather than chase that, the index was
  rewritten as a plain `(N, 384)` NumPy matrix with a single matmul per
  query — which is also just the *correct* choice at course scale (a few
  thousand chunks), since an ANN index only starts paying for its complexity
  past ~100K+ vectors. Swappable behind the same `VectorStore` interface if
  a deployment ever needs to scale past that.
- **The explanation agent is instructed to under-claim, not over-claim** —
  it cites evidence and hedges, never asserts misconduct. A wrongly
  confident false accusation is worse than a missed detection.
- **Granite backend is swappable** (`GRANITE_BACKEND=watsonx` vs. `local`) so
  the project runs both on the mandated IBM Cloud Lite / watsonx.ai path and
  fully offline for CI/local dev.

## Evaluation

Unlike a hand-wavy "it seems to work" plagiarism tool, each signal is
evaluated independently against a public benchmark before being trusted in
the fused score. Full numbers in `eval/results/*.json`; summary:

| Signal | Dataset | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **AI-text detector** | [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (700 human vs. ChatGPT answers, held-out test split) | 0.91 | 0.99 | **0.95** | **0.99** |
| **Semantic similarity** — realistic task | [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) paraphrases vs. randomly-paired unrelated text (488 pairs) | 0.99 | 0.98 | **0.99** | **1.00** |
| Semantic similarity — adversarial task | PAWS's own hard-negative labels (600 pairs, same words/different meaning) | 0.41 | 0.98 | 0.58 | 0.63 |
| **Stylometric drift** | [ELI5 authorship-attribution](https://huggingface.co/datasets/manu/eli5_authorship_attribution), 150 authors, same- vs. cross-author (~750-word pseudo-documents) | 0.56 | 0.91 | 0.69 | 0.65 |

Two results are worth explaining rather than glossing over:

- **Similarity: two very different numbers for the same model, on purpose.**
  PAWS's own task is deliberately adversarial - "the dog bit the man" vs.
  "the man bit the dog" share every word but mean the opposite thing, and a
  generic (not paraphrase-fine-tuned) sentence embedding is *expected* to
  struggle there (F1 0.58 - a known, documented limitation, not a bug). The
  "realistic" row re-evaluates the same model on the case that actually
  matters here - genuine paraphrase vs. ordinary unrelated text, i.e. what a
  copied-and-reworded paragraph vs. a different student's different essay
  actually looks like - and it's excellent (F1 0.99). Reporting both, rather
  than only the flattering one, is the point: it shows *where* the signal is
  strong and *where* it isn't, instead of one cherry-picked number.
- **Stylometric drift is the weakest of the three signals (F1 0.69, ROC-AUC
  0.65 vs. 0.95-1.00 for the other two) - and that's reflected in the
  design, not hidden from it.** `RiskFusionModel`'s default weights give it
  the lowest coefficient (1.0, vs. 3.2 for similarity and 2.4 for
  AI-detection - see `app/fusion/risk_model.py`), because style is
  inherently noisier than semantic content: it needs long samples (500+
  words; classic authorship-attribution literature bears this out) and
  several prior submissions before it's reliable at all
  (`StyleProfile.DriftResult.is_reliable`). It's kept in the system anyway
  because it's the *only* one of the three signals that can catch
  contract-cheating/ghostwriting - text that matches nothing online and
  isn't AI-generated, so the other two are structurally blind to it -
  making it valuable as corroborating evidence even though it's not strong
  enough to be a standalone detector.

Reproduce with:
```bash
python -m eval.scripts.train_ai_detector
python -m eval.scripts.eval_similarity_detector
python -m eval.scripts.eval_stylometry_detector
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # fill in watsonx credentials, or leave GRANITE_BACKEND=local

uvicorn app.api.main:app --reload          # terminal 1
streamlit run dashboard/app.py             # terminal 2
```

Or with Docker: `docker compose up --build` (see `deploy/DEPLOY.md` for IBM
Cloud Code Engine deployment).

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
dashboard/      Streamlit instructor console
eval/           evaluation scripts + results (real, reproducible numbers)
deploy/         Dockerfiles + IBM Cloud deployment guide
tests/          unit tests for the detection core
```

## Honest limitations / future work

- The AI-text detector is trained on HC3, which skews toward Q&A-style text;
  a course with a very different genre (creative writing, code comments)
  would benefit from fine-tuning on in-domain examples.
- Style-drift confidence is low until a student has 3+ prior submissions on
  file — by design (see `StyleProfile.DriftResult.is_reliable`), but it does
  mean the first assignment of a semester gets weaker signal.
- The fusion model currently caches per-course in-process memory
  (`_fusion_model_cache` in `app/api/deps.py`); a multi-worker production
  deployment would move that to Redis or reload-from-disk per request.
- No web-search fallback yet (checking against sources *outside* the course,
  not just other students' submissions) — would slot in as an additional
  LangGraph tool node without changing the fusion/agent architecture.

## Credits

Built by Nayana S as the capstone project for the **IBM SkillsBuild for
University Engagements — AICTE 2026** Agentic AI internship, in
collaboration with Edunet Foundation.
