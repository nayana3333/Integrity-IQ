from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import Base, engine

from . import routes_auth, routes_courses, routes_feedback, routes_submissions


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="IntegrityIQ",
    description="Adaptive, multi-signal academic-integrity assistant (RAG-free "
    "semantic similarity + AI-text detection + stylometric drift, fused and "
    "explained by an IBM Granite agent).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_auth.router)
app.include_router(routes_courses.router)
app.include_router(routes_submissions.router)
app.include_router(routes_feedback.router)


@app.get("/health")
def health():
    return {"status": "ok"}
