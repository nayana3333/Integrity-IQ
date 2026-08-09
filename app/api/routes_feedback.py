from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agents.feedback_loop import FeedbackLoopAgent
from app.db import get_db
from app.db.models import Instructor, InstructorVerdict, Submission

from . import deps
from .auth import get_current_instructor
from .routes_courses import _get_owned_course
from .schemas import FeedbackCreate, RetrainResult

router = APIRouter(tags=["feedback"])


@router.post("/submissions/{submission_id}/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(
    submission_id: str,
    payload: FeedbackCreate,
    db: Session = Depends(get_db),
    instructor: Instructor = Depends(get_current_instructor),
):
    if payload.verdict not in ("confirmed", "false_positive"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "verdict must be 'confirmed' or 'false_positive'")

    submission = db.get(Submission, submission_id)
    if submission is None or submission.course.instructor_id != instructor.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")

    existing = db.query(InstructorVerdict).filter_by(submission_id=submission_id).one_or_none()
    if existing:
        existing.verdict = payload.verdict
        existing.notes = payload.notes
    else:
        db.add(
            InstructorVerdict(
                submission_id=submission_id,
                instructor_id=instructor.id,
                verdict=payload.verdict,
                notes=payload.notes,
            )
        )
    db.commit()
    return {"status": "recorded"}


@router.post("/courses/{course_id}/retrain", response_model=RetrainResult)
def retrain_course_model(
    course_id: str,
    db: Session = Depends(get_db),
    instructor: Instructor = Depends(get_current_instructor),
):
    course = _get_owned_course(course_id, db, instructor)

    labeled: list[tuple[dict, str]] = []
    for submission in course.submissions:
        if submission.verdict is not None and submission.components:
            labeled.append((submission.components, submission.verdict.verdict))

    agent = FeedbackLoopAgent(get_fusion_model=deps.get_fusion_model)
    result = agent.retrain_course(course_id, labeled)
    return RetrainResult(**result)
