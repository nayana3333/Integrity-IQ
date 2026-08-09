"""The adaptive half of the system: turns accumulated instructor verdicts
(confirmed / false-positive) into training data for that course's
`RiskFusionModel`, so the fusion weights - how much to trust similarity vs.
AI-detection vs. style-drift - are learned per course from real feedback
instead of staying fixed forever.

This directly implements the "adaptive AI system that learns from
historical assignment submissions and instructor feedback" requirement from
the original problem statement, and is what separates this from a static
plagiarism scanner: two courses with very different assignment styles (a
creative-writing course vs. a lab-report course, say) end up with
differently calibrated models because their instructors' feedback differs.
"""
from __future__ import annotations

from app.fusion import RiskFusionModel, FEATURE_NAMES


class FeedbackLoopAgent:
    def __init__(self, get_fusion_model):
        self._get_fusion_model = get_fusion_model

    def retrain_course(
        self, course_id: str, labeled_components: list[tuple[dict[str, float], str]]
    ) -> dict:
        """labeled_components: list of (components_dict, verdict) where verdict
        is "confirmed" or "false_positive", pulled from InstructorVerdict rows
        joined against the SimilarityFlag/Submission components snapshot taken
        at analysis time.
        """
        X = [[c[name] for name in FEATURE_NAMES] for c, _ in labeled_components]
        y = [1 if verdict == "confirmed" else 0 for _, verdict in labeled_components]

        model: RiskFusionModel = self._get_fusion_model(course_id)
        return model.retrain(X, y)
