from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, EmailStr


class InstructorCreate(BaseModel):
    email: EmailStr
    password: str
    name: str


class InstructorOut(BaseModel):
    id: str
    email: EmailStr
    name: str

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CourseCreate(BaseModel):
    code: str
    name: str


class CourseOut(BaseModel):
    id: str
    code: str
    name: str

    model_config = {"from_attributes": True}


class StudentCreate(BaseModel):
    external_id: str
    name: str


class StudentOut(BaseModel):
    id: str
    external_id: str
    name: str

    model_config = {"from_attributes": True}


class SimilarityFlagOut(BaseModel):
    chunk_index: int
    query_text: str
    matched_student_id: str
    matched_text: str
    similarity: float

    model_config = {"from_attributes": True}


class SubmissionOut(BaseModel):
    id: str
    student_id: str
    filename: str
    submitted_at: dt.datetime
    status: str
    risk_probability: float | None
    severity: str | None
    components: dict | None
    explanation: str | None
    similarity_flags: list[SimilarityFlagOut] = []

    model_config = {"from_attributes": True}


class FeedbackCreate(BaseModel):
    verdict: str  # "confirmed" | "false_positive"
    notes: str | None = None


class RetrainResult(BaseModel):
    trained: bool
    reason: str | None = None
    n_samples: int | None = None
    learned_weights: dict[str, float] | None = None
