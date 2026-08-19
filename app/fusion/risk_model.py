"""Fuses the three independent detection signals into one calibrated,
per-submission risk score - and, critically, *learns* how to weight them
per course from instructor feedback instead of using fixed hand-tuned
weights forever.

Why fusion instead of three separate flags: each signal alone is weak.
Semantic similarity alone flags legitimate common phrasing shared by an
entire class. Perplexity alone is fooled by a student who writes plainly.
Style drift alone is noisy for students with only one or two prior
submissions on file. Combined, and calibrated against real instructor
verdicts, they're far more reliable than any one of them - this is the
same "ensemble of weak, cheap signals beats one strong, expensive one"
idea used in production fraud-detection systems.

Cold start: a brand-new course has no feedback yet, so `RiskFusionModel`
falls back to a fixed, documented default weighting until at least
`MIN_LABELS_TO_TRAIN` instructor verdicts have been collected for that
course, at which point `retrain()` swaps in a learned logistic regression.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ["max_similarity", "flagged_chunk_ratio", "ai_proba", "style_drift"]

# Fixed default weights (logistic-regression-style: sigmoid(w . x + b)),
# hand-set from the relative reliability of each signal for a cold-start
# course with zero labeled feedback. Deliberately conservative (biased
# toward not over-flagging) until real feedback lets us calibrate per course.
_DEFAULT_WEIGHTS = np.array([3.2, 1.8, 2.4, 1.0])
_DEFAULT_BIAS = -2.6

MIN_LABELS_TO_TRAIN = 25


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskReport:
    submission_id: str
    probability: float
    severity: Severity
    components: dict[str, float]
    is_calibrated: bool  # True once a per-course learned model is in use


def _severity_from_probability(p: float) -> Severity:
    if p >= 0.85:
        return Severity.CRITICAL
    if p >= 0.6:
        return Severity.HIGH
    if p >= 0.3:
        return Severity.MEDIUM
    return Severity.LOW


class RiskFusionModel:
    """One instance per course. Persists to disk under
    `models/fusion/{course_id}.joblib` once trained.
    """

    def __init__(self, course_id: str, models_dir: str | Path = "models/fusion"):
        self.course_id = course_id
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = self.models_dir / f"{course_id}.joblib"
        self._clf: LogisticRegression | None = None
        if self._model_path.exists():
            self._clf = joblib.load(self._model_path)

    @property
    def is_calibrated(self) -> bool:
        return self._clf is not None

    def score(
        self,
        submission_id: str,
        max_similarity: float,
        flagged_chunk_ratio: float,
        ai_proba: float,
        style_drift: float,
    ) -> RiskReport:
        x = np.array([[max_similarity, flagged_chunk_ratio, ai_proba, style_drift]])

        if self._clf is not None:
            probability = float(self._clf.predict_proba(x)[0][1])
        else:
            z = float(x[0] @ _DEFAULT_WEIGHTS + _DEFAULT_BIAS)
            probability = 1.0 / (1.0 + math.exp(-z))

        components = dict(zip(FEATURE_NAMES, x[0].tolist(), strict=True))
        return RiskReport(
            submission_id=submission_id,
            probability=probability,
            severity=_severity_from_probability(probability),
            components=components,
            is_calibrated=self.is_calibrated,
        )

    def retrain(self, X: list[list[float]], y: list[int]) -> dict:
        """Retrain on accumulated instructor feedback for this course.

        X rows are [max_similarity, flagged_chunk_ratio, ai_proba, style_drift]
        y is 1 if the instructor confirmed the flag as genuine misconduct,
        0 if they marked it a false positive.
        """
        if len(y) < MIN_LABELS_TO_TRAIN:
            return {
                "trained": False,
                "reason": f"Need {MIN_LABELS_TO_TRAIN} labeled verdicts, have {len(y)}.",
            }
        if len(set(y)) < 2:
            return {"trained": False, "reason": "Need both positive and negative examples."}

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(np.array(X), np.array(y))
        self._clf = clf
        joblib.dump(clf, self._model_path)

        return {
            "trained": True,
            "n_samples": len(y),
            "learned_weights": dict(zip(FEATURE_NAMES, clf.coef_[0].tolist(), strict=True)),
        }
