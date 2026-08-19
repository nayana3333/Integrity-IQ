import tempfile

from app.fusion import FEATURE_NAMES, RiskFusionModel, Severity
from app.fusion.risk_model import MIN_LABELS_TO_TRAIN


def test_cold_start_uses_default_weights():
    with tempfile.TemporaryDirectory() as tmp:
        model = RiskFusionModel("course-1", models_dir=tmp)
        assert not model.is_calibrated

        low_risk = model.score("s1", max_similarity=0.1, flagged_chunk_ratio=0.0, ai_proba=0.1, style_drift=0.2)
        high_risk = model.score("s2", max_similarity=0.95, flagged_chunk_ratio=0.8, ai_proba=0.9, style_drift=3.0)

        assert low_risk.severity == Severity.LOW
        assert high_risk.severity in (Severity.HIGH, Severity.CRITICAL)
        assert high_risk.probability > low_risk.probability


def test_retrain_requires_minimum_labels():
    with tempfile.TemporaryDirectory() as tmp:
        model = RiskFusionModel("course-2", models_dir=tmp)
        result = model.retrain(X=[[0.5, 0.5, 0.5, 0.5]], y=[1])
        assert result["trained"] is False


def test_retrain_learns_and_persists():
    with tempfile.TemporaryDirectory() as tmp:
        model = RiskFusionModel("course-3", models_dir=tmp)

        X, y = [], []
        for _ in range(MIN_LABELS_TO_TRAIN // 2 + 1):
            X.append([0.95, 0.9, 0.9, 3.0])  # clearly risky
            y.append(1)
            X.append([0.1, 0.0, 0.05, 0.1])  # clearly clean
            y.append(0)

        result = model.retrain(X, y)
        assert result["trained"] is True
        assert model.is_calibrated
        assert set(result["learned_weights"].keys()) == set(FEATURE_NAMES)

        # A fresh instance pointed at the same dir should load the persisted model.
        reloaded = RiskFusionModel("course-3", models_dir=tmp)
        assert reloaded.is_calibrated
