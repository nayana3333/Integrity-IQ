"""Instructor-facing dashboard for IntegrityIQ.

A thin Streamlit client over the FastAPI backend - deliberately has zero
detection logic of its own, it only calls the API and renders what comes
back. Keeping the UI dumb like this means the same backend could grow a
React frontend later without touching app/ at all.

Visual layer lives in `theme.py` (CSS injection + small HTML component
helpers) so this file stays about data flow, not markup.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from theme import empty_state, header, inject_css, severity_badge_html, stat_card

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="IntegrityIQ", page_icon="🛡️", layout="wide")
inject_css()


def api(method: str, path: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state['token']}"
    return requests.request(method, f"{API_BASE_URL}{path}", headers=headers, **kwargs)


def require_login():
    st.markdown('<div class="iq-auth-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="iq-auth-brand">
            <div class="iq-mark">🛡️</div>
            <h1>IntegrityIQ</h1>
            <p>Adaptive academic-integrity assistant — instructor console</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        tab_login, tab_register = st.tabs(["Log in", "Register"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="you@school.edu")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if submitted:
                resp = api("POST", "/auth/login", data={"username": email, "password": password})
                if resp.status_code == 200:
                    st.session_state["token"] = resp.json()["access_token"]
                    st.session_state["email"] = email
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Login failed"))

        with tab_register:
            with st.form("register_form"):
                name = st.text_input("Name", placeholder="Jane Doe")
                reg_email = st.text_input("Email", key="reg_email", placeholder="you@school.edu")
                reg_password = st.text_input(
                    "Password", type="password", key="reg_password", placeholder="••••••••"
                )
                reg_submitted = st.form_submit_button(
                    "Create account", type="primary", use_container_width=True
                )
            if reg_submitted:
                resp = api(
                    "POST",
                    "/auth/register",
                    json={"name": name, "email": reg_email, "password": reg_password},
                )
                if resp.status_code == 201:
                    st.success("Account created — switch to the Log in tab.")
                else:
                    st.error(resp.json().get("detail", "Registration failed"))

    st.markdown("</div>", unsafe_allow_html=True)


def _select_with_sticky_default(
    label: str, options: dict[str, dict], session_key: str, create_label: str
) -> str:
    """A selectbox that re-selects the item just created/picked after a
    st.rerun(), instead of silently resetting to `create_label` every time -
    Streamlit selectboxes have no memory of their own across reruns, so
    without this the sidebar looked broken (create succeeds, but the new
    course/student never appears selected) even though the backend call
    worked fine.
    """
    labels = [create_label, *options.keys()]
    default_index = 0
    selected_id = st.session_state.get(session_key)
    if selected_id:
        for i, item_label in enumerate(labels):
            if item_label != create_label and options[item_label]["id"] == selected_id:
                default_index = i
                break
    return st.selectbox(label, labels, index=default_index, label_visibility="collapsed")


def course_picker() -> dict | None:
    resp = api("GET", "/courses")
    courses = resp.json() if resp.status_code == 200 else []

    with st.sidebar:
        st.markdown("**:material/school: Course**")
        options = {f"{c['code']} — {c['name']}": c for c in courses}
        choice = _select_with_sticky_default(
            "Select a course", options, "selected_course_id", "+ New course"
        )

        if choice == "+ New course":
            with st.container(border=True):
                code = st.text_input("Course code", placeholder="CS301", key="new_course_code")
                name = st.text_input("Course name", placeholder="Intro to Algorithms", key="new_course_name")
                if st.button("Create course", type="primary", use_container_width=True) and code and name:
                    r = api("POST", "/courses", json={"code": code, "name": name})
                    if r.status_code == 201:
                        st.session_state["selected_course_id"] = r.json()["id"]
                        st.session_state.pop("selected_student_id", None)
                        st.rerun()
                    else:
                        st.error(r.text)
            return None

        selected = options[choice]
        st.session_state["selected_course_id"] = selected["id"]
        return selected


def student_picker(course: dict) -> dict | None:
    resp = api("GET", f"/courses/{course['id']}/students")
    students = resp.json() if resp.status_code == 200 else []

    with st.sidebar:
        st.markdown("**:material/person: Student**")
        options = {f"{s['name']} ({s['external_id']})": s for s in students}
        choice = _select_with_sticky_default(
            "Select a student", options, "selected_student_id", "+ New student"
        )

        if choice == "+ New student":
            with st.container(border=True):
                ext_id = st.text_input("Roll number / ID", placeholder="T001", key="new_student_id")
                name = st.text_input("Full name", placeholder="Alex Rivera", key="new_student_name")
                if st.button("Add student", type="primary", use_container_width=True) and ext_id and name:
                    r = api(
                        "POST",
                        f"/courses/{course['id']}/students",
                        json={"external_id": ext_id, "name": name},
                    )
                    if r.status_code == 201:
                        st.session_state["selected_student_id"] = r.json()["id"]
                        st.rerun()
                    else:
                        st.error(r.text)
            return None

        selected = options[choice]
        st.session_state["selected_student_id"] = selected["id"]
        return selected


def tab_upload(course: dict, student: dict | None):
    if student is None:
        empty_state("👤", "Pick or add a student in the sidebar to start an integrity check.")
        return

    with st.container(border=True):
        assignment_title = st.text_input("Assignment title", value="Assignment 1")
        uploaded = st.file_uploader(
            "Submission file", type=["pdf", "docx", "txt"], label_visibility="visible"
        )
        run = st.button(
            ":material/search_insights: Run integrity check",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

    if uploaded and run:
        with st.spinner(
            "Running similarity search, AI-detection, style analysis, and generating the report…"
        ):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            resp = api(
                "POST",
                f"/courses/{course['id']}/students/{student['id']}/submissions",
                files=files,
                params={"assignment_title": assignment_title},
            )
        if resp.status_code != 201:
            st.error(resp.text)
            return

        st.markdown("#### Result")
        render_submission_report(resp.json())


def render_submission_report(result: dict):
    comps = result.get("components") or {}
    st.markdown(
        '<div class="iq-stats">'
        + stat_card("Risk score", f"{result['risk_probability']:.0%}")
        + stat_card("Severity", severity_badge_html(result["severity"]))
        + stat_card("Max similarity", f"{comps.get('max_similarity', 0):.0%}")
        + stat_card("AI-text likelihood", f"{comps.get('ai_proba', 0):.0%}")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="iq-panel">
            <div class="iq-panel-label">📄 Instructor report</div>
            {result.get('explanation') or '<em>No report generated.</em>'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    flags = result.get("similarity_flags") or []
    if flags:
        st.markdown(f"**:material/flag: Flagged passages** &nbsp;·&nbsp; {len(flags)} match(es)")
        for flag in flags:
            with st.expander(f"Chunk {flag['chunk_index']} — {flag['similarity']:.0%} similarity"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="iq-diff-label">This submission</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="iq-diff-text">{flag["query_text"]}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        f'<div class="iq-diff-label">Matched student ({flag["matched_student_id"][:8]}…)</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<div class="iq-diff-text">{flag["matched_text"]}</div>', unsafe_allow_html=True)


def tab_flagged(course: dict):
    resp = api("GET", f"/courses/{course['id']}/submissions")
    submissions = resp.json() if resp.status_code == 200 else []
    submissions = sorted(submissions, key=lambda s: s.get("risk_probability") or 0, reverse=True)

    if not submissions:
        empty_state("📭", "No submissions analyzed yet — upload one in the first tab.")
        return

    for sub in submissions:
        badge = severity_badge_html(sub["severity"])
        label = f"{sub['filename']}  ·  {sub.get('risk_probability', 0):.0%} risk"
        with st.expander(label):
            st.markdown(badge, unsafe_allow_html=True)
            render_submission_report(sub)
            c1, c2 = st.columns(2)
            if c1.button(
                ":material/check_circle: Confirm misconduct", key=f"confirm_{sub['id']}", use_container_width=True
            ):
                api("POST", f"/submissions/{sub['id']}/feedback", json={"verdict": "confirmed"})
                st.success("Recorded.")
            if c2.button(":material/cancel: False positive", key=f"fp_{sub['id']}", use_container_width=True):
                api("POST", f"/submissions/{sub['id']}/feedback", json={"verdict": "false_positive"})
                st.success("Recorded.")


def tab_retrain(course: dict):
    with st.container(border=True):
        st.markdown("##### :material/model_training: Adaptive risk model")
        st.caption(
            "Once enough submissions have a confirmed/false-positive verdict, retraining "
            "recalibrates how much this course's model trusts each signal — semantic "
            "similarity, AI-text detection, and stylometric drift."
        )
        if st.button("Retrain now", type="primary"):
            resp = api("POST", f"/courses/{course['id']}/retrain")
            result = resp.json()
            if result.get("trained"):
                st.success(f"Retrained on {result['n_samples']} labeled submissions.")
                st.json(result["learned_weights"])
            else:
                st.warning(result.get("reason", "Not enough data to retrain yet."))


_PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, sans-serif", "color": "#9099ab"},
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
}


def tab_style_trends(course: dict):
    resp = api("GET", f"/courses/{course['id']}/submissions")
    submissions = resp.json() if resp.status_code == 200 else []
    rows = [
        {
            "submitted_at": s["submitted_at"],
            "filename": s["filename"],
            "style_drift": (s.get("components") or {}).get("style_drift"),
            "student_id": s["student_id"],
        }
        for s in submissions
        if (s.get("components") or {}).get("style_drift") is not None
    ]
    if not rows:
        empty_state("📈", "No style-drift data yet.")
        return

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x="submitted_at", y="style_drift", color="student_id", hover_data=["filename"],
        color_discrete_sequence=["#6366f1", "#f472b6", "#22d3ee", "#facc15", "#4ade80"],
    )
    fig.update_traces(marker={"size": 11, "line": {"width": 1, "color": "#0b0e14"}})
    fig.update_layout(**_PLOTLY_LAYOUT, xaxis_title=None, yaxis_title="Drift score")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig, use_container_width=True)


def main():
    if not st.session_state.get("token"):
        require_login()
        return

    email = st.session_state.get("email", "")
    initial = (email[:1] or "?").upper()
    header(
        "Instructor console",
        right_html=(
            f'<span class="iq-chip">'
            f'<span style="width:20px;height:20px;border-radius:50%;background:var(--iq-primary);'
            f'display:inline-flex;align-items:center;justify-content:center;font-size:11px;'
            f'font-weight:700;color:white;">{initial}</span>{email}</span>'
        ),
    )

    with st.sidebar:
        if st.button(":material/logout: Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.divider()

    course = course_picker()
    if course is None:
        empty_state("🏫", "Create or select a course in the sidebar to get started.")
        return

    student = student_picker(course)

    tabs = st.tabs([
        ":material/upload_file: Upload & Analyze",
        ":material/flag: Flagged Submissions",
        ":material/show_chart: Style Trends",
        ":material/model_training: Retrain Model",
    ])
    with tabs[0]:
        tab_upload(course, student)
    with tabs[1]:
        tab_flagged(course)
    with tabs[2]:
        tab_style_trends(course)
    with tabs[3]:
        tab_retrain(course)


if __name__ == "__main__":
    main()
