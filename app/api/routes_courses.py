from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import Course, Instructor, Student

from .auth import get_current_instructor
from .schemas import CourseCreate, CourseOut, StudentCreate, StudentOut

router = APIRouter(prefix="/courses", tags=["courses"])


def _get_owned_course(course_id: str, db: Session, instructor: Instructor) -> Course:
    course = db.get(Course, course_id)
    if course is None or course.instructor_id != instructor.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Course not found")
    return course


@router.post("", response_model=CourseOut, status_code=status.HTTP_201_CREATED)
def create_course(
    payload: CourseCreate,
    db: Session = Depends(get_db),
    instructor: Instructor = Depends(get_current_instructor),
):
    course = Course(code=payload.code, name=payload.name, instructor_id=instructor.id)
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


@router.get("", response_model=list[CourseOut])
def list_courses(
    db: Session = Depends(get_db), instructor: Instructor = Depends(get_current_instructor)
):
    return db.query(Course).filter_by(instructor_id=instructor.id).all()


@router.post("/{course_id}/students", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
def add_student(
    course_id: str,
    payload: StudentCreate,
    db: Session = Depends(get_db),
    instructor: Instructor = Depends(get_current_instructor),
):
    course = _get_owned_course(course_id, db, instructor)
    student = Student(external_id=payload.external_id, name=payload.name, course_id=course.id)
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


@router.get("/{course_id}/students", response_model=list[StudentOut])
def list_students(
    course_id: str,
    db: Session = Depends(get_db),
    instructor: Instructor = Depends(get_current_instructor),
):
    course = _get_owned_course(course_id, db, instructor)
    return course.students
