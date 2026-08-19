"""Evaluates the stylometric drift signal (app.stylometry.StyleProfile) on
the ELI5 authorship-attribution dataset (manu/eli5_authorship_attribution on
Hugging Face - 1.5M Reddit r/explainlikeimfive comments labeled with a real
per-author user ID, not a demographic bucket): for each of N authors with
enough comments, we build a baseline from a few of their comments, then
score the drift of (a) a held-out sample from that SAME author (should score
low - the "genuine, no ghostwriting" case) against (b) a sample from a
DIFFERENT random author (should score high - this simulates a
ghostwritten/contract-cheating submission, which is exactly the case
web-similarity search can't catch because the ghostwritten text won't match
anything online).

Individual Reddit comments are far shorter than a real assignment (median
~40 words vs. hundreds), and classic stylometry/authorship-attribution
literature (Burrows' Delta etc.) generally needs 500+ word samples for
function-word-frequency features to stabilize - so raw single comments were
tried first and, as expected, performed close to chance (ROC-AUC ~0.55-0.60,
see git history / eval_ai_detector_run.log for that first pass). Each
"document" here is therefore CONCAT_N raw comments concatenated together,
which approximates assignment-length text without fabricating data - it's
still 100% real author text, just enough of it to make word-frequency
features meaningful, matching the length regime the signal is designed for.

Usage:
    python -m eval.scripts.eval_stylometry_detector [--n-authors 150] [--concat-n 6]

Writes:
    eval/results/stylometry_detector_metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.stylometry import StyleProfile

RESULTS_DIR = Path("eval/results")
MIN_WORDS = 30
BASELINE_DOCS = 3


def load_author_comments(min_comments: int, max_rows: int = 600_000) -> dict[str, list[str]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="manu/eli5_authorship_attribution", repo_type="dataset", filename="eli5_authors.csv"
    )
    df = pd.read_csv(path, usecols=["user", "text"], nrows=max_rows, low_memory=False)
    df = df.dropna(subset=["user", "text"])
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]

    by_author: dict[str, list[str]] = defaultdict(list)
    for user, text in zip(df["user"], df["text"], strict=True):
        by_author[str(user)].append(text)

    return {aid: posts for aid, posts in by_author.items() if len(posts) >= min_comments}


def make_pseudo_docs(comments: list[str], n_docs: int, concat_n: int, rng: random.Random) -> list[str]:
    comments = list(comments)
    rng.shuffle(comments)
    return [" ".join(comments[i * concat_n : (i + 1) * concat_n]) for i in range(n_docs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-authors", type=int, default=150)
    parser.add_argument("--concat-n", type=int, default=6, help="raw comments concatenated per pseudo-document")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    needed_comments = (BASELINE_DOCS + 1) * args.concat_n

    print(f"Loading ELI5 comments and grouping by author (need >= {needed_comments} qualifying "
          f"comments/author to build {BASELINE_DOCS} baseline + 1 held-out pseudo-document "
          f"of {args.concat_n} concatenated comments each)...")
    by_author = load_author_comments(min_comments=needed_comments)
    author_ids = list(by_author.keys())
    rng.shuffle(author_ids)
    author_ids = author_ids[: args.n_authors]
    print(f"Using {len(author_ids)} authors, out of {len(by_author)} qualifying.")

    y_true: list[int] = []
    drift_scores: list[float] = []
    pseudo_doc_word_counts: list[int] = []

    for i, author_id in enumerate(author_ids):
        docs = make_pseudo_docs(by_author[author_id], BASELINE_DOCS + 1, args.concat_n, rng)
        baseline_docs, held_out_same = docs[:BASELINE_DOCS], docs[BASELINE_DOCS]
        pseudo_doc_word_counts.append(len(held_out_same.split()))

        other_author_id = rng.choice([a for a in author_ids if a != author_id])
        held_out_diff = make_pseudo_docs(by_author[other_author_id], 1, args.concat_n, rng)[0]

        profile = StyleProfile(student_id=author_id)
        for doc in baseline_docs:
            profile.update(doc)

        same_author_result = profile.score(held_out_same)
        diff_author_result = profile.score(held_out_diff)

        y_true += [0, 1]
        drift_scores += [same_author_result.drift_score, diff_author_result.drift_score]

        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(author_ids)} authors")

    y_true = np.array(y_true)
    drift_scores = np.array(drift_scores)

    thresholds = np.linspace(drift_scores.min(), drift_scores.max(), 200)
    best_f1, best_t = 0.0, thresholds[0]
    for t in thresholds:
        pred = (drift_scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    y_pred = (drift_scores >= best_t).astype(int)

    metrics = {
        "n_authors": len(author_ids),
        "n_comparisons": len(y_true),
        "concat_n_comments_per_doc": args.concat_n,
        "avg_words_per_pseudo_document": float(np.mean(pseudo_doc_word_counts)),
        "best_threshold": float(best_t),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": float(best_f1),
        "roc_auc": roc_auc_score(y_true, drift_scores),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "dataset": "manu/eli5_authorship_attribution (eli5_authors.csv, sampled, comments "
        "concatenated to approximate assignment-length documents)",
        "baseline_docs_per_author": BASELINE_DOCS,
        "note": "This signal is intentionally weighted lowest of the three in RiskFusionModel's "
        "default weights - it's a real but noisier signal than semantic similarity or "
        "AI-text detection, used as corroborating evidence (esp. for ghostwriting, which the "
        "other two signals structurally cannot catch) rather than a standalone detector.",
    }

    print(json.dumps(metrics, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "stylometry_detector_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_DIR / 'stylometry_detector_metrics.json'}")


if __name__ == "__main__":
    main()
