from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import Instructor

from .auth import create_access_token, hash_password, verify_password
from .schemas import InstructorCreate, InstructorOut, Token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=InstructorOut, status_code=status.HTTP_201_CREATED)
def register(payload: InstructorCreate, db: Session = Depends(get_db)):
    existing = db.query(Instructor).filter_by(email=payload.email).one_or_none()
    if existing:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")

    instructor = Instructor(
        email=payload.email,
        name=payload.name,
        hashed_password=hash_password(payload.password),
    )
    db.add(instructor)
    db.commit()
    db.refresh(instructor)
    return instructor


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    instructor = db.query(Instructor).filter_by(email=form.username).one_or_none()
    if not instructor or not verify_password(form.password, instructor.hashed_password):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=instructor.id)
    return Token(access_token=token)
