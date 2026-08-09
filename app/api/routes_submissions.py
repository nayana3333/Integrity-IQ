from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.agents.orchestrator import run_integrity_check
from app.db import get_db
from app.db.models import Instructor, SimilarityFlag, Student, Submission
from app.ingestion.parser import extract_text

from . import deps
from .auth import get_current_instructor
from .routes_courses import _get_owned_course
from .schemas import SubmissionOut

router = APIRouter(tags=["submissions"])


@router.post(
    "/courses/{course_id}/students/{student_id}/submissions",
    response_model=SubmissionOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_submission(
    course_id: str,
    student_id: str,
    file: UploadFile,
    assignment_title: str = "Untitled assignment",
    db: Session = Depends(get_db),
    instructor: Instructor = Depends(get_current_instructor),
):
    course = _get_owned_course(course_id, db, instructor)
    student = db.get(Student, student_id)
    if student is None or student.course_id != course.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found in this course")

    raw_bytes = await file.read()
    try:
        text = extract_text(raw_bytes, file.filename or "submission.txt")
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    submission = Submission(
        course_id=course.id,
        student_id=student.id,
        filename=file.filename or "submission.txt",
        raw_text=text,
        status="pending",
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    pipeline = deps.make_pipeline(db)
    explainer = deps.get_explanation_agent()

    try:
        state = run_integrity_check(
            pipeline,
            explainer,
            course_id=course.id,
            submission_id=submission.id,
            student_id=student.id,
            student_label=f"{student.name} ({student.external_id})",
            assignment_title=assignment_title,
            text=text,
        )
    except Exception as exc:
        submission.status = "error"
        db.commit()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Analysis failed: {exc}") from exc

    result = state["detection_result"]
    submission.status = "processed"
    submission.risk_probability = result.risk_report.probability
    submission.severity = result.risk_report.severity.value
    submission.components = result.risk_report.components
    submission.explanation = state["report"]
    db.add(submission)

    for chunk_idx, matches in result.similarity_matches.items():
        for m in matches:
            db.add(
                SimilarityFlag(
                    submission_id=submission.id,
                    chunk_index=chunk_idx,
                    query_text=m.query_text,
                    matched_submission_id=m.matched_submission_id,
                    matched_student_id=m.matched_student_id,
                    matched_chunk_index=m.matched_chunk_index,
                    matched_text=m.matched_text,
                    similarity=m.similarity,
                )
            )
    db.commit()
    db.refresh(submission)
    return submission


@router.get("/courses/{course_id}/submissions", response_model=list[SubmissionOut])
def list_submissions(
    course_id: str,
    db: Session = Depends(get_db),
    instructor: Instructor = Depends(get_current_instructor),
):
    course = _get_owned_course(course_id, db, instructor)
    return course.submissions


@router.get("/submissions/{submission_id}", response_model=SubmissionOut)
def get_submission(
    submission_id: str,
    db: Session = Depends(get_db),
    instructor: Instructor = Depends(get_current_instructor),
):
    submission = db.get(Submission, submission_id)
    if submission is None or submission.course.instructor_id != instructor.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    return submission
