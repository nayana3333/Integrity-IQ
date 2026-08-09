"""Trained classifier on top of the perplexity/burstiness/lexical features.

Ships with a small, transparent heuristic fallback (a hand-set sigmoid over
mean perplexity) so the pipeline runs end-to-end before you've trained on a
labeled dataset. Once you run `eval/scripts/train_ai_detector.py` (which
trains on the HC3 human-vs-ChatGPT dataset), a calibrated logistic
regression model is loaded instead - see eval/results/ai_detector_metrics.json
for the real precision/recall numbers after training.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import joblib
import numpy as np

from .features import AI_FEATURE_NAMES, extract_ai_features

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "ai_detector.joblib"


class AITextDetector:
    def __init__(self, model_path: str | Path = _DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._pipeline = None
        if self.model_path.exists():
            self._pipeline = joblib.load(self.model_path)
        else:
            warnings.warn(
                f"No trained AI-detector model found at {self.model_path}. "
                "Falling back to an untrained perplexity heuristic - run "
                "eval/scripts/train_ai_detector.py for calibrated results.",
                stacklevel=2,
            )

    def predict_proba(self, text: str) -> tuple[float, dict[str, float]]:
        """Returns (probability_ai_generated, raw_features)."""
        features = extract_ai_features(text)
        vector = np.array([[features[name] for name in AI_FEATURE_NAMES]])

        if self._pipeline is not None:
            proba = float(self._pipeline.predict_proba(vector)[0][1])
        else:
            proba = _heuristic_proba(features)

        return proba, features


def _heuristic_proba(features: dict[str, float]) -> float:
    """Untrained fallback: lower perplexity + lower burstiness => more
    likely AI-generated. Calibrated loosely against typical GPT-2 perplexity
    ranges (human prose: ~40-80, fluent LLM output: ~10-30) - a rough prior,
    not a substitute for the trained classifier.
    """
    perplexity = features["mean_perplexity"]
    burstiness = features["burstiness"]
    score = -0.06 * (perplexity - 35) - 0.8 * (burstiness - 1.2)
    return 1.0 / (1.0 + math.exp(-score))
