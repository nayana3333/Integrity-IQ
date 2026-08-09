"""Turns raw detection signals into a human-readable, evidence-cited report
for the instructor - the actual "agent" step that makes this more than a
scoring script.

Framing matters a lot here: the prompt explicitly instructs Granite to
(1) stay evidence-based and cite specific matched passages instead of just
asserting a verdict, (2) use hedged, non-accusatory language ("the
similarity below warrants review", not "this student cheated"), and
(3) always end with the confidence/reliability caveat instructor should
weigh (few historical samples => less reliable style-drift signal). This is
a deliberate, defensible design choice, not an afterthought: a wrongly
confident false accusation is a far worse failure mode here than a missed
detection, so the agent is instructed to under-claim rather than over-claim.

If Granite is unavailable (no credentials, model not downloaded, network
down), falls back to a deterministic template - the report is blander but
the pipeline never breaks because an LLM call failed.
"""
from __future__ import annotations

from app.fusion import Severity
from app.pipeline import DetectionResult

from .llm_client import GraniteClient

_SYSTEM_PROMPT = """You are an academic-integrity assistant that writes short, \
evidence-based reports for instructors reviewing a flagged assignment submission. \
Rules you must follow:
- Never assert the student cheated. Describe evidence and its strength; let the \
instructor decide.
- Cite specific matched passages or style deviations, don't just state a score.
- If the underlying signal has low confidence (e.g. few prior submissions on file), \
say so explicitly.
- Keep it to 4-6 sentences, plain language, no headers.
"""


class ExplanationAgent:
    def __init__(self, client: GraniteClient | None = None):
        self._client = client or GraniteClient()

    def generate_report(
        self,
        result: DetectionResult,
        student_label: str,
        assignment_title: str,
    ) -> str:
        prompt = self._build_prompt(result, student_label, assignment_title)
        try:
            return self._client.generate(f"{_SYSTEM_PROMPT}\n\n{prompt}")
        except Exception:
            return self._template_fallback(result, student_label, assignment_title)

    def _build_prompt(self, result: DetectionResult, student_label: str, assignment_title: str) -> str:
        top_matches = sorted(
            (m for matches in result.similarity_matches.values() for m in matches),
            key=lambda m: m.similarity,
            reverse=True,
        )[:3]
        matches_text = "\n".join(
            f"- {m.similarity:.0%} semantic match with another student's submission: "
            f'"{m.query_text[:150]}..." <-> "{m.matched_text[:150]}..."'
            for m in top_matches
        ) or "- No high-similarity passages found."

        return f"""Assignment: {assignment_title}
Student: {student_label}
Overall risk score: {result.risk_report.probability:.0%} ({result.risk_report.severity.value})

Semantic similarity evidence:
{matches_text}

AI-generated-text likelihood: {result.ai_proba:.0%}

Writing-style drift from student's own baseline: {result.style_drift.explain()}

Write the instructor-facing report now."""

    def _template_fallback(
        self, result: DetectionResult, student_label: str, assignment_title: str
    ) -> str:
        severity_phrase = {
            Severity.LOW: "shows no significant indicators of concern",
            Severity.MEDIUM: "shows some indicators worth a quick manual look",
            Severity.HIGH: "shows multiple indicators that warrant review",
            Severity.CRITICAL: "shows strong, multi-signal indicators that warrant prompt review",
        }[result.risk_report.severity]

        n_matches = sum(len(v) for v in result.similarity_matches.values())
        return (
            f"{student_label}'s submission for '{assignment_title}' {severity_phrase} "
            f"(overall score {result.risk_report.probability:.0%}). "
            f"{n_matches} passage(s) matched other students' work at high semantic similarity, "
            f"AI-generated-text likelihood was estimated at {result.ai_proba:.0%}, and "
            f"{result.style_drift.explain().lower()} "
            f"This is an automated signal, not a verdict - please review the highlighted "
            f"passages before taking any action."
        )
