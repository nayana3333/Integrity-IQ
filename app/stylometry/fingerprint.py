"""Per-student writing-style fingerprints with online (streaming) updates.

We keep a running mean/variance per feature per student using Welford's
algorithm instead of storing every past essay - this keeps the profile
O(1) in space regardless of submission history length, and lets us update
it incrementally as new (confirmed-original) submissions come in.

Drift is scored as a diagonal Mahalanobis distance (sum of per-feature
squared z-scores). A full-covariance Mahalanobis distance would be more
precise, but needs far more samples than a single course ever produces
(70+ features vs. a handful of essays per student) - the diagonal
approximation is the standard practical compromise for small-sample
stylometry and is explicit about that tradeoff rather than hiding it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .features import FEATURE_NAMES, extract_style_vector


@dataclass
class StyleProfile:
    student_id: str
    n_samples: int = 0
    mean: dict[str, float] = field(default_factory=lambda: dict.fromkeys(FEATURE_NAMES, 0.0))
    _m2: dict[str, float] = field(default_factory=lambda: dict.fromkeys(FEATURE_NAMES, 0.0))

    @property
    def variance(self) -> dict[str, float]:
        if self.n_samples < 2:
            return dict.fromkeys(FEATURE_NAMES, 0.0)
        return {f: self._m2[f] / (self.n_samples - 1) for f in FEATURE_NAMES}

    def update(self, text: str) -> dict[str, float]:
        """Welford's online mean/variance update. Returns the extracted vector."""
        vec = extract_style_vector(text)
        self.n_samples += 1
        for f in FEATURE_NAMES:
            x = vec[f]
            delta = x - self.mean[f]
            self.mean[f] += delta / self.n_samples
            delta2 = x - self.mean[f]
            self._m2[f] += delta * delta2
        return vec

    def score(self, text: str, min_samples_for_confidence: int = 3) -> DriftResult:
        vec = extract_style_vector(text)
        variance = self.variance
        z_scores: dict[str, float] = {}
        for f in FEATURE_NAMES:
            var = variance[f]
            std = math.sqrt(var) if var > 1e-6 else 1.0
            z_scores[f] = (vec[f] - self.mean[f]) / std

        # Diagonal Mahalanobis-style distance, normalized by sqrt(n_features)
        # so the score sits in a roughly comparable range regardless of how
        # many features we're tracking.
        raw = math.sqrt(sum(z * z for z in z_scores.values()) / len(FEATURE_NAMES))

        top_contributors = sorted(
            z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:5]

        confidence = min(1.0, self.n_samples / min_samples_for_confidence)

        return DriftResult(
            student_id=self.student_id,
            drift_score=raw,
            confidence=confidence,
            n_baseline_samples=self.n_samples,
            top_contributors=top_contributors,
        )

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "n_samples": self.n_samples,
            "mean": self.mean,
            "m2": self._m2,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StyleProfile:
        profile = cls(student_id=data["student_id"], n_samples=data["n_samples"])
        profile.mean = data["mean"]
        profile._m2 = data["m2"]
        return profile


@dataclass
class DriftResult:
    student_id: str
    drift_score: float
    confidence: float  # 0-1, low when baseline has too few samples to trust
    n_baseline_samples: int
    top_contributors: list[tuple[str, float]]

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 1.0

    def explain(self) -> str:
        if not self.is_reliable:
            return (
                f"Only {self.n_baseline_samples} prior submission(s) on file for "
                f"this student - drift score is informational only, not yet reliable."
            )
        parts = []
        for name, z in self.top_contributors[:3]:
            label = f'use of "{name[3:]}"' if name.startswith("fw_") else name
            direction = "higher" if z > 0 else "lower"
            parts.append(f"{label} ({direction} than usual, z={z:.1f})")
        return "Largest deviations from this student's usual writing: " + "; ".join(parts)
