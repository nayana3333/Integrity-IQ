"""Instructor-facing dashboard for IntegrityIQ.

A thin Streamlit client over the FastAPI backend - deliberately has zero
detection logic of its own, it only calls the API and renders what comes
back. Keeping the UI dumb like this means the same backend could grow a
React frontend later without touching app/ at all.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="IntegrityIQ", page_icon="🛡️", layout="wide")


def api(method: str, path: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state['token']}"
    return requests.request(method, f"{API_BASE_URL}{path}", headers=headers, **kwargs)


def require_login():
    st.title("🛡️ IntegrityIQ")
    st.caption("Adaptive academic-integrity assistant — instructor console")

    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            resp = api("POST", "/auth/login", data={"username": email, "password": password})
            if resp.status_code == 200:
                st.session_state["token"] = resp.json()["access_token"]
                st.rerun()
            else:
                st.error(resp.json().get("detail", "Login failed"))

    with tab_register:
        with st.form("register_form"):
            name = st.text_input("Name")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_submitted = st.form_submit_button("Create account")
        if reg_submitted:
            resp = api(
                "POST",
                "/auth/register",
                json={"name": name, "email": reg_email, "password": reg_password},
            )
            if resp.status_code == 201:
                st.success("Account created — log in on the other tab.")
            else:
                st.error(resp.json().get("detail", "Registration failed"))


def severity_badge(severity: str | None) -> str:
    colors = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
    return f"{colors.get(severity, '⚪')} {(severity or 'n/a').upper()}"


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
    labels = [create_label] + list(options.keys())
    default_index = 0
    selected_id = st.session_state.get(session_key)
    if selected_id:
        for i, item_label in enumerate(labels):
            if item_label != create_label and options[item_label]["id"] == selected_id:
                default_index = i
                break
    return st.selectbox(label, labels, index=default_index)


def course_picker() -> dict | None:
    resp = api("GET", "/courses")
    courses = resp.json() if resp.status_code == 200 else []

    with st.sidebar:
        st.subheader("Course")
        options = {f"{c['code']} — {c['name']}": c for c in courses}
        choice = _select_with_sticky_default(
            "Select a course", options, "selected_course_id", "<create new>"
        )

        if choice == "<create new>":
            with st.form("new_course"):
                code = st.text_input("Course code (e.g. CS301)")
                name = st.text_input("Course name")
                if st.form_submit_button("Create course") and code and name:
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
        st.subheader("Student")
        options = {f"{s['name']} ({s['external_id']})": s for s in students}
        choice = _select_with_sticky_default(
            "Select a student", options, "selected_student_id", "<add new>"
        )

        if choice == "<add new>":
            with st.form("new_student"):
                ext_id = st.text_input("Roll number / ID")
                name = st.text_input("Full name")
                if st.form_submit_button("Add student") and ext_id and name:
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
    st.header("Upload & analyze a submission")
    if student is None:
        st.info("Pick or add a student in the sidebar first.")
        return

    assignment_title = st.text_input("Assignment title", value="Assignment 1")
    uploaded = st.file_uploader("Submission file", type=["pdf", "docx", "txt"])

    if uploaded and st.button("Run integrity check", type="primary"):
        with st.spinner("Running similarity search, AI-detection, style analysis, and generating the report..."):
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

        result = resp.json()
        render_submission_report(result)


def render_submission_report(result: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk score", f"{result['risk_probability']:.0%}")
    col2.metric("Severity", severity_badge(result["severity"]))
    comps = result.get("components") or {}
    col3.metric("Max similarity match", f"{comps.get('max_similarity', 0):.0%}")

    st.subheader("Instructor report")
    st.write(result.get("explanation") or "_No report generated._")

    flags = result.get("similarity_flags") or []
    if flags:
        st.subheader(f"Flagged passages ({len(flags)})")
        for flag in flags:
            with st.expander(f"Chunk {flag['chunk_index']} — {flag['similarity']:.0%} similarity"):
                c1, c2 = st.columns(2)
                c1.markdown("**This submission**")
                c1.write(flag["query_text"])
                c2.markdown(f"**Matched student ({flag['matched_student_id'][:8]}…)**")
                c2.write(flag["matched_text"])


def tab_flagged(course: dict):
    st.header("Flagged submissions")
    resp = api("GET", f"/courses/{course['id']}/submissions")
    submissions = resp.json() if resp.status_code == 200 else []
    submissions = sorted(submissions, key=lambda s: s.get("risk_probability") or 0, reverse=True)

    if not submissions:
        st.info("No submissions analyzed yet.")
        return

    for sub in submissions:
        label = f"{severity_badge(sub['severity'])} — {sub['filename']} — {sub.get('risk_probability', 0):.0%}"
        with st.expander(label):
            render_submission_report(sub)
            c1, c2 = st.columns(2)
            if c1.button("✅ Confirm misconduct", key=f"confirm_{sub['id']}"):
                api("POST", f"/submissions/{sub['id']}/feedback", json={"verdict": "confirmed"})
                st.success("Recorded.")
            if c2.button("❌ False positive", key=f"fp_{sub['id']}"):
                api("POST", f"/submissions/{sub['id']}/feedback", json={"verdict": "false_positive"})
                st.success("Recorded.")


def tab_retrain(course: dict):
    st.header("Adaptive model — retrain from instructor feedback")
    st.write(
        "Once enough submissions have a confirmed/false-positive verdict, "
        "retraining recalibrates how much this course's model trusts each "
        "signal (similarity vs. AI-detection vs. style drift)."
    )
    if st.button("Retrain now", type="primary"):
        resp = api("POST", f"/courses/{course['id']}/retrain")
        result = resp.json()
        if result.get("trained"):
            st.success(f"Retrained on {result['n_samples']} labeled submissions.")
            st.json(result["learned_weights"])
        else:
            st.warning(result.get("reason", "Not enough data to retrain yet."))


def tab_style_trends(course: dict):
    st.header("Style-drift trend")
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
        st.info("No data yet.")
        return

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x="submitted_at", y="style_drift", color="student_id", hover_data=["filename"],
        title="Writing-style drift over time, per student",
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    if not st.session_state.get("token"):
        require_login()
        return

    with st.sidebar:
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

    course = course_picker()
    if course is None:
        st.info("Create or select a course in the sidebar to get started.")
        return

    student = student_picker(course)

    tabs = st.tabs(["Upload & Analyze", "Flagged Submissions", "Style Trends", "Retrain Model"])
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
