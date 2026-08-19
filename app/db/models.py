from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class Instructor(Base):
    __tablename__ = "instructors"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)

    courses: Mapped[list[Course]] = relationship(back_populates="instructor")


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    code: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String)
    instructor_id: Mapped[str] = mapped_column(ForeignKey("instructors.id"))

    instructor: Mapped[Instructor] = relationship(back_populates="courses")
    students: Mapped[list[Student]] = relationship(back_populates="course")
    submissions: Mapped[list[Submission]] = relationship(back_populates="course")


class Student(Base):
    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    external_id: Mapped[str] = mapped_column(String, index=True)  # roll number etc.
    name: Mapped[str] = mapped_column(String)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"))

    course: Mapped[Course] = relationship(back_populates="students")
    submissions: Mapped[list[Submission]] = relationship(back_populates="student")
    style_profile: Mapped[StyleProfileRecord | None] = relationship(
        back_populates="student", uselist=False
    )


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"))
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    filename: Mapped[str] = mapped_column(String)
    raw_text: Mapped[str] = mapped_column(Text)
    submitted_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    status: Mapped[str] = mapped_column(String, default="pending")  # pending|processed|error
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[str | None] = mapped_column(String, nullable=True)
    components: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    course: Mapped[Course] = relationship(back_populates="submissions")
    student: Mapped[Student] = relationship(back_populates="submissions")
    similarity_flags: Mapped[list[SimilarityFlag]] = relationship(back_populates="submission")
    verdict: Mapped[InstructorVerdict | None] = relationship(
        back_populates="submission", uselist=False
    )


class SimilarityFlag(Base):
    __tablename__ = "similarity_flags"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id"))
    chunk_index: Mapped[int] = mapped_column()
    query_text: Mapped[str] = mapped_column(Text)

    matched_submission_id: Mapped[str] = mapped_column(String)
    matched_student_id: Mapped[str] = mapped_column(String)
    matched_chunk_index: Mapped[int] = mapped_column()
    matched_text: Mapped[str] = mapped_column(Text)
    similarity: Mapped[float] = mapped_column(Float)

    submission: Mapped[Submission] = relationship(back_populates="similarity_flags")


class InstructorVerdict(Base):
    __tablename__ = "instructor_verdicts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id"), unique=True)
    instructor_id: Mapped[str] = mapped_column(ForeignKey("instructors.id"))
    verdict: Mapped[str] = mapped_column(String)  # "confirmed" | "false_positive"
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    submission: Mapped[Submission] = relationship(back_populates="verdict")


class StyleProfileRecord(Base):
    __tablename__ = "style_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), unique=True)
    n_samples: Mapped[int] = mapped_column(default=0)
    mean_json: Mapped[dict] = mapped_column(JSON, default=dict)
    m2_json: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    student: Mapped[Student] = relationship(back_populates="style_profile")
