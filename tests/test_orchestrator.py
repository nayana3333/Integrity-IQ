"""Integration test for the LangGraph detect->explain orchestration.

Uses a fake GraniteClient (no network/model download) so this stays fast and
deterministic - the real watsonx/local Granite backends are exercised
manually (see deploy/DEPLOY.md), not in the unit-test suite, since they
need either cloud credentials or a multi-gigabyte local model download.
"""
import tempfile

from app.agents.explain import ExplanationAgent
from app.agents.orchestrator import run_integrity_check
from app.ai_detector import AITextDetector
from app.embeddings import VectorStore
from app.fusion import RiskFusionModel
from app.pipeline import DetectionPipeline
from app.stylometry import StyleProfile


class FakeGraniteClient:
    def generate(self, prompt: str, max_new_tokens: int = 400, temperature: float = 0.2) -> str:
        return "[stubbed Granite report] " + prompt.splitlines()[-1]


def test_orchestrator_runs_detect_then_explain():
    with tempfile.TemporaryDirectory() as vec_dir, tempfile.TemporaryDirectory() as fusion_dir:
        vector_store = VectorStore(persist_dir=vec_dir)
        ai_detector = AITextDetector(model_path="nonexistent.joblib")

        profiles: dict[str, StyleProfile] = {}

        def load_profile(student_id: str) -> StyleProfile:
            return profiles.get(student_id, StyleProfile(student_id=student_id))

        def save_profile(profile: StyleProfile) -> None:
            profiles[profile.student_id] = profile

        def get_fusion_model(course_id: str) -> RiskFusionModel:
            return RiskFusionModel(course_id, models_dir=fusion_dir)

        pipeline = DetectionPipeline(
            vector_store=vector_store,
            ai_detector=ai_detector,
            load_profile=load_profile,
            save_profile=save_profile,
            get_fusion_model=get_fusion_model,
        )
        explainer = ExplanationAgent(client=FakeGraniteClient())

        text = (
            "The mitochondria is the organelle responsible for producing "
            "ATP through cellular respiration. It has a double membrane "
            "structure that increases surface area for chemical reactions."
        )

        state = run_integrity_check(
            pipeline,
            explainer,
            course_id="course-1",
            submission_id="sub-1",
            student_id="student-1",
            student_label="Test Student (T001)",
            assignment_title="Biology HW 3",
            text=text,
        )

        assert state["detection_result"] is not None
        assert 0.0 <= state["detection_result"].risk_report.probability <= 1.0
        assert state["report"].startswith("[stubbed Granite report]")
