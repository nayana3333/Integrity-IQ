"""Process-wide singletons (embedding model, vector store, AI detector) plus
DB-backed adapters that satisfy the callback interfaces `DetectionPipeline`
expects. Keeping these adapters here - rather than inside `app.pipeline` -
is what keeps the pipeline module itself framework/DB-agnostic.
"""
from __future__ import annotations

import functools

from sqlalchemy.orm import Session

from app.agents.explain import ExplanationAgent
from app.ai_detector import AITextDetector
from app.db import models
from app.embeddings import VectorStore
from app.fusion import RiskFusionModel
from app.pipeline import DetectionPipeline
from app.stylometry import StyleProfile


@functools.lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(persist_dir="vector_store_data")


@functools.lru_cache(maxsize=1)
def get_ai_detector() -> AITextDetector:
    return AITextDetector()


@functools.lru_cache(maxsize=1)
def get_explanation_agent() -> ExplanationAgent:
    return ExplanationAgent()


_fusion_models: dict[str, RiskFusionModel] = {}


def get_fusion_model(course_id: str) -> RiskFusionModel:
    if course_id not in _fusion_models:
        _fusion_models[course_id] = RiskFusionModel(course_id)
    return _fusion_models[course_id]


def make_pipeline(db: Session) -> DetectionPipeline:
    def load_profile(student_id: str) -> StyleProfile:
        record = db.get(models.StyleProfileRecord, student_id, options=[])
        record = (
            db.query(models.StyleProfileRecord)
            .filter_by(student_id=student_id)
            .one_or_none()
        )
        if record is None:
            return StyleProfile(student_id=student_id)
        return StyleProfile.from_dict(
            {
                "student_id": student_id,
                "n_samples": record.n_samples,
                "mean": record.mean_json,
                "m2": record.m2_json,
            }
        )

    def save_profile(profile: StyleProfile) -> None:
        record = (
            db.query(models.StyleProfileRecord)
            .filter_by(student_id=profile.student_id)
            .one_or_none()
        )
        if record is None:
            record = models.StyleProfileRecord(student_id=profile.student_id)
            db.add(record)
        record.n_samples = profile.n_samples
        record.mean_json = profile.mean
        record.m2_json = profile._m2
        db.commit()

    return DetectionPipeline(
        vector_store=get_vector_store(),
        ai_detector=get_ai_detector(),
        load_profile=load_profile,
        save_profile=save_profile,
        get_fusion_model=get_fusion_model,
    )
