"""Wires the three independent detection signals together into one
`DetectionPipeline.analyze()` call.

Deliberately decoupled from the database and the web framework (it takes
plain callbacks for loading/saving a student's style profile) so the whole
detection core can be unit-tested and reused - by the FastAPI backend, by
the agent tools in app.agents, and by the offline evaluation harness in
eval/ - without needing a running server or database.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.ai_detector import AITextDetector
from app.embeddings import VectorStore, SimilarityMatch
from app.fusion import RiskFusionModel, RiskReport
from app.ingestion.parser import chunk_text
from app.stylometry import DriftResult, StyleProfile

ProfileLoader = Callable[[str], StyleProfile]
ProfileSaver = Callable[[StyleProfile], None]


@dataclass
class DetectionResult:
    submission_id: str
    risk_report: RiskReport
    similarity_matches: dict[int, list[SimilarityMatch]]
    ai_proba: float
    ai_features: dict[str, float]
    style_drift: DriftResult


class DetectionPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        ai_detector: AITextDetector,
        load_profile: ProfileLoader,
        save_profile: ProfileSaver,
        get_fusion_model: Callable[[str], RiskFusionModel],
    ):
        self._vector_store = vector_store
        self._ai_detector = ai_detector
        self._load_profile = load_profile
        self._save_profile = save_profile
        self._get_fusion_model = get_fusion_model

    def analyze(
        self,
        course_id: str,
        submission_id: str,
        student_id: str,
        text: str,
        sentences_per_chunk: int = 5,
        overlap: int = 2,
    ) -> DetectionResult:
        chunks = chunk_text(text, sentences_per_chunk=sentences_per_chunk, overlap=overlap)

        # 1. Semantic similarity against every other submission in the course.
        similarity_matches = self._vector_store.find_similar(
            chunks, course_id=course_id, exclude_submission_id=submission_id
        )
        max_similarity = max(
            (m.similarity for matches in similarity_matches.values() for m in matches),
            default=0.0,
        )
        flagged_chunk_ratio = len(similarity_matches) / len(chunks) if chunks else 0.0

        # 2. AI-generated-text likelihood.
        ai_proba, ai_features = self._ai_detector.predict_proba(text)

        # 3. Stylometric drift from this student's own baseline.
        profile = self._load_profile(student_id)
        style_drift = profile.score(text)

        # 4. Fuse into one calibrated risk score.
        fusion_model = self._get_fusion_model(course_id)
        risk_report = fusion_model.score(
            submission_id=submission_id,
            max_similarity=max_similarity,
            flagged_chunk_ratio=flagged_chunk_ratio,
            ai_proba=ai_proba,
            style_drift=min(style_drift.drift_score, 5.0) * style_drift.confidence,
        )

        # 5. Index this submission's chunks for future comparisons, and fold
        # it into the student's style baseline. NOTE: this optimistically
        # assumes the submission is genuine; if an instructor later confirms
        # misconduct, call `exclude_from_baseline` (see app.db integration)
        # to avoid the baseline being poisoned by ghostwritten text.
        self._vector_store.add_submission(course_id, submission_id, student_id, chunks)
        profile.update(text)
        self._save_profile(profile)

        return DetectionResult(
            submission_id=submission_id,
            risk_report=risk_report,
            similarity_matches=similarity_matches,
            ai_proba=ai_proba,
            ai_features=ai_features,
            style_drift=style_drift,
        )
