"""Evaluates the semantic-similarity signal (app.embeddings.VectorStore) two
ways, on PAWS (Paraphrase Adversaries from Word Scrambling):

1. "adversarial" - PAWS's actual designed task: distinguish genuine
   paraphrases (label 1) from PAWS's specifically-constructed hard negatives
   (label 0), which share almost all the same words but differ in meaning
   via word-order/structure changes (e.g. "the dog bit the man" vs "the man
   bit the dog"). This is a known hard case for plain sentence-embedding
   cosine similarity - PAWS was built specifically to expose it, and a
   generic (not paraphrase-fine-tuned) embedding model is expected to score
   modestly here. We report it anyway rather than hide it: it's an honest,
   documented limitation (see README "Honest limitations").

2. "realistic" - closer to what the system actually needs to get right in
   production: distinguish a genuine paraphrase (PAWS label-1 pairs) from
   ordinary, topically-unrelated text (a randomly-paired different PAWS
   sentence, not an adversarially-constructed near-miss). Real plagiarism
   cases and real non-matches look like this case far more often than the
   adversarial one - a copied-and-reworded paragraph vs. a completely
   different student's completely different essay - so this second number
   is the more representative one for the actual use case.

Usage:
    python -m eval.scripts.eval_similarity_detector [--n 600] [--threshold 0.80]

Writes:
    eval/results/similarity_detector_metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.embeddings.vector_store import get_embedder

RESULTS_DIR = Path("eval/results")


def load_paws_pairs(n: int, seed: int = 42):
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return list(ds["sentence1"]), list(ds["sentence2"]), list(ds["label"])


def _score_pairs(embedder, s1: list[str], s2: list[str]) -> np.ndarray:
    e1 = embedder.encode(s1, normalize_embeddings=True)
    e2 = embedder.encode(s2, normalize_embeddings=True)
    return np.sum(e1 * e2, axis=1)  # cosine similarity (vectors are normalized)


def _evaluate(y: np.ndarray, similarities: np.ndarray, threshold: float) -> dict:
    y_pred = (similarities >= threshold).astype(int)

    precisions, recalls, thresholds = precision_recall_curve(y, similarities)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = int(np.argmax(f1s))

    return {
        "n_pairs": len(y),
        "positive_rate": float(y.mean()),
        "threshold_used": threshold,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, similarities),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "best_possible_threshold": float(thresholds[best_idx]) if best_idx < len(thresholds) else threshold,
        "best_possible_f1": float(f1s[best_idx]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    print(f"Loading {args.n} PAWS pairs...")
    s1, s2, labels = load_paws_pairs(args.n, seed=args.seed)
    y_adversarial = np.array(labels)

    embedder = get_embedder()

    print("Scoring the adversarial task (PAWS's own labels)...")
    adversarial_similarities = _score_pairs(embedder, s1, s2)
    adversarial_metrics = _evaluate(y_adversarial, adversarial_similarities, args.threshold)

    print("Building and scoring the realistic task (paraphrase vs. unrelated text)...")
    paraphrase_idx = [i for i, label in enumerate(labels) if label == 1]
    pos_s1 = [s1[i] for i in paraphrase_idx]
    pos_s2 = [s2[i] for i in paraphrase_idx]

    shuffled = list(range(len(pos_s1)))
    rng.shuffle(shuffled)
    # Pair each sentence1 with a DIFFERENT random pair's sentence2 -> almost
    # certainly a topically unrelated sentence, i.e. what a real non-match
    # looks like (not an adversarial near-miss).
    neg_s1 = pos_s1
    neg_s2 = [pos_s2[j] for j in shuffled]

    realistic_s1 = pos_s1 + neg_s1
    realistic_s2 = pos_s2 + neg_s2
    y_realistic = np.array([1] * len(pos_s1) + [0] * len(neg_s1))

    realistic_similarities = _score_pairs(embedder, realistic_s1, realistic_s2)
    realistic_metrics = _evaluate(y_realistic, realistic_similarities, args.threshold)

    metrics = {
        "embedding_model": "all-MiniLM-L6-v2",
        "dataset": "google-research-datasets/paws (labeled_final, test split)",
        "adversarial_task": {
            "description": "PAWS's own label: genuine paraphrase vs. constructed hard negative "
            "(same words, different meaning). A generic embedding model is expected to "
            "struggle here - see README.",
            **adversarial_metrics,
        },
        "realistic_task": {
            "description": "Genuine paraphrase (PAWS label=1) vs. randomly-paired unrelated "
            "sentence - representative of real plagiarism-detection non-matches.",
            **realistic_metrics,
        },
    }

    print(json.dumps(metrics, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "similarity_detector_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_DIR / 'similarity_detector_metrics.json'}")


if __name__ == "__main__":
    main()
