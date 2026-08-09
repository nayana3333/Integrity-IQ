"""Trains the AI-generated-text classifier on the HC3 dataset (Hello-SimpleAI/HC3),
which pairs the same questions answered by humans and by ChatGPT across
multiple domains (medicine, finance, open-domain QA, etc.) - a good proxy
for "could a student have submitted this as their own answer".

Usage:
    python -m eval.scripts.train_ai_detector [--n-per-class 400]

Writes:
    models/ai_detector.joblib               <- trained sklearn pipeline
    eval/results/ai_detector_metrics.json   <- precision/recall/F1/AUC on a
                                                held-out split, for the README
                                                and resume.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ai_detector.features import AI_FEATURE_NAMES, extract_ai_features

MODELS_DIR = Path("models")
RESULTS_DIR = Path("eval/results")


def load_hc3_examples(n_per_class: int, seed: int = 42) -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    # HC3's original loading script is no longer supported by recent
    # `datasets` versions (script-based datasets were deprecated for
    # security reasons); load the auto-converted parquet mirror instead,
    # which HF generates for every script-based dataset under this ref.
    ds = load_dataset(
        "Hello-SimpleAI/HC3", "default", split="train", revision="refs/convert/parquet"
    )
    rng = random.Random(seed)

    human_texts: list[str] = []
    ai_texts: list[str] = []
    for row in ds:
        for ans in row.get("human_answers") or []:
            if ans and len(ans.split()) >= 25:
                human_texts.append(ans)
        for ans in row.get("chatgpt_answers") or []:
            if ans and len(ans.split()) >= 25:
                ai_texts.append(ans)

    rng.shuffle(human_texts)
    rng.shuffle(ai_texts)
    human_texts = human_texts[:n_per_class]
    ai_texts = ai_texts[:n_per_class]

    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)
    return texts, labels


def extract_features_batch(texts: list[str]) -> np.ndarray:
    rows = []
    start = time.time()
    for i, text in enumerate(texts):
        feats = extract_ai_features(text)
        rows.append([feats[name] for name in AI_FEATURE_NAMES])
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  extracted {i + 1}/{len(texts)} ({elapsed:.0f}s elapsed)")
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=400)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    print(f"Loading up to {args.n_per_class} human + {args.n_per_class} ChatGPT examples from HC3...")
    texts, labels = load_hc3_examples(args.n_per_class)
    print(f"Loaded {len(texts)} examples ({sum(labels)} AI, {len(labels) - sum(labels)} human).")

    print("Extracting perplexity/burstiness features (this calls GPT-2 per sentence, "
          "so it's the slow part)...")
    X = extract_features_batch(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_names": AI_FEATURE_NAMES,
        "learned_coefficients": dict(
            zip(AI_FEATURE_NAMES, pipeline.named_steps["clf"].coef_[0].tolist())
        ),
        "dataset": "Hello-SimpleAI/HC3 (config=all)",
    }

    print(json.dumps(metrics, indent=2))

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(pipeline, MODELS_DIR / "ai_detector.joblib")
    with open(RESULTS_DIR / "ai_detector_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved trained model to {MODELS_DIR / 'ai_detector.joblib'}")
    print(f"Saved metrics to {RESULTS_DIR / 'ai_detector_metrics.json'}")


if __name__ == "__main__":
    main()
