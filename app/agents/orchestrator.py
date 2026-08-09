"""LangGraph orchestration for a single integrity check: detect -> explain.

Kept intentionally small (two nodes) rather than padded out for its own
sake - the graph structure is what it needs to be for the current feature
set. It's built as a graph rather than a plain function call chain because
the natural next steps (see README "Future Work") - e.g. an "escalate for
deeper web search if similarity is borderline", or a "route to a
course-specific explanation style" branch - are graph nodes/conditional
edges away, not a rewrite.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.pipeline import DetectionPipeline, DetectionResult

from .explain import ExplanationAgent


class IntegrityCheckState(TypedDict):
    course_id: str
    submission_id: str
    student_id: str
    student_label: str
    assignment_title: str
    text: str
    detection_result: Optional[DetectionResult]
    report: Optional[str]


def _build_graph(pipeline: DetectionPipeline, explainer: ExplanationAgent):
    graph = StateGraph(IntegrityCheckState)

    def detect_node(state: IntegrityCheckState) -> dict:
        result = pipeline.analyze(
            course_id=state["course_id"],
            submission_id=state["submission_id"],
            student_id=state["student_id"],
            text=state["text"],
        )
        return {"detection_result": result}

    def explain_node(state: IntegrityCheckState) -> dict:
        result: DetectionResult = state["detection_result"]
        report = explainer.generate_report(
            result, state["student_label"], state["assignment_title"]
        )
        return {"report": report}

    graph.add_node("detect", detect_node)
    graph.add_node("explain", explain_node)
    graph.set_entry_point("detect")
    graph.add_edge("detect", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


def run_integrity_check(
    pipeline: DetectionPipeline,
    explainer: ExplanationAgent,
    *,
    course_id: str,
    submission_id: str,
    student_id: str,
    student_label: str,
    assignment_title: str,
    text: str,
) -> IntegrityCheckState:
    compiled_graph = _build_graph(pipeline, explainer)
    return compiled_graph.invoke(
        {
            "course_id": course_id,
            "submission_id": submission_id,
            "student_id": student_id,
            "student_label": student_label,
            "assignment_title": assignment_title,
            "text": text,
            "detection_result": None,
            "report": None,
        }
    )
