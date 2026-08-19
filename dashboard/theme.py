"""Design system for the IntegrityIQ dashboard.

Streamlit's default widgets are functional but generic - this module layers
a small, consistent design system on top (type scale, color tokens, card
components, severity badges) via one CSS injection plus a handful of HTML
snippet builders, rather than fighting Streamlit widget-by-widget. Colors
are duplicated between here and `.streamlit/config.toml` on purpose: the
config.toml values set Streamlit's own native widget theme (inputs,
buttons, base background), while these CSS variables extend that same
palette to the custom card/badge components below - keeping both in sync
by hand is simpler than templating one from the other for a design system
this size.
"""
from __future__ import annotations

import streamlit as st

SEVERITY_COLORS = {
    "low": {"fg": "#4ade80", "bg": "rgba(74, 222, 128, 0.12)", "border": "#4ade80"},
    "medium": {"fg": "#fbbf24", "bg": "rgba(251, 191, 36, 0.12)", "border": "#fbbf24"},
    "high": {"fg": "#fb923c", "bg": "rgba(251, 146, 60, 0.14)", "border": "#fb923c"},
    "critical": {"fg": "#f87171", "bg": "rgba(248, 113, 113, 0.14)", "border": "#f87171"},
}
_DEFAULT_SEVERITY = {"fg": "#9ca3af", "bg": "rgba(156, 163, 175, 0.12)", "border": "#9ca3af"}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --iq-bg: #0b0e14;
            --iq-surface: #141922;
            --iq-surface-2: #1a2029;
            --iq-border: rgba(255,255,255,0.08);
            --iq-border-strong: rgba(255,255,255,0.14);
            --iq-text: #e6e8ee;
            --iq-text-dim: #9099ab;
            --iq-text-faint: #626b7d;
            --iq-primary: #6366f1;
            --iq-primary-dim: #4f52d6;
            --iq-primary-tint: rgba(99, 102, 241, 0.12);
            --iq-radius-sm: 8px;
            --iq-radius: 12px;
            --iq-radius-lg: 16px;
            --iq-shadow: 0 1px 2px rgba(0,0,0,0.24), 0 8px 24px -12px rgba(0,0,0,0.5);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        code, .iq-mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }

        .stApp { background: var(--iq-bg); }
        #MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }

        section[data-testid="stSidebar"] {
            background: var(--iq-surface);
            border-right: 1px solid var(--iq-border);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

        .block-container { padding-top: 2rem; max-width: 1100px; }

        /* Headings */
        h1, h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em; }
        h2 { font-size: 1.35rem !important; margin-top: 0.25rem !important; }

        /* Tabs -> pill-style segmented control */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: var(--iq-surface-2);
            padding: 4px;
            border-radius: var(--iq-radius);
            border: 1px solid var(--iq-border);
        }
        .stTabs [data-baseweb="tab"] {
            height: 38px;
            border-radius: var(--iq-radius-sm);
            padding: 0 16px;
            color: var(--iq-text-dim);
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: var(--iq-primary) !important;
            color: white !important;
        }
        .stTabs [data-baseweb="tab-highlight"] { display: none; }
        .stTabs [data-baseweb="tab-border"] { display: none; }

        /* Buttons */
        .stButton > button {
            border-radius: var(--iq-radius-sm);
            font-weight: 600;
            border: 1px solid var(--iq-border-strong);
            transition: transform 0.06s ease, border-color 0.15s ease;
        }
        .stButton > button:hover { border-color: var(--iq-primary); transform: translateY(-1px); }
        .stButton > button[kind="primary"] {
            background: var(--iq-primary);
            border: 1px solid var(--iq-primary);
        }
        .stButton > button[kind="primary"]:hover { background: var(--iq-primary-dim); }

        /* Inputs */
        .stTextInput input, .stSelectbox [data-baseweb="select"] > div, .stTextArea textarea {
            border-radius: var(--iq-radius-sm) !important;
            background: var(--iq-surface-2) !important;
            border: 1px solid var(--iq-border-strong) !important;
        }

        /* Native bordered containers -> elevated cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--iq-radius) !important;
            border-color: var(--iq-border) !important;
            background: var(--iq-surface);
        }

        /* Expanders -> cards */
        div[data-testid="stExpander"] {
            border-radius: var(--iq-radius) !important;
            border-color: var(--iq-border) !important;
            background: var(--iq-surface);
            box-shadow: var(--iq-shadow);
            overflow: hidden;
        }
        div[data-testid="stExpander"] summary {
            padding: 0.85rem 1rem !important;
            font-weight: 500;
        }

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--iq-surface-2) !important;
            border: 1.5px dashed var(--iq-border-strong) !important;
            border-radius: var(--iq-radius) !important;
        }

        hr { border-color: var(--iq-border) !important; }

        /* ---- custom components ---- */
        .iq-header {
            display: flex; align-items: center; gap: 14px;
            padding-bottom: 1.25rem; margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--iq-border);
        }
        .iq-header .iq-mark {
            width: 42px; height: 42px; border-radius: 11px;
            background: linear-gradient(145deg, var(--iq-primary), #8b5cf6);
            display: flex; align-items: center; justify-content: center;
            font-size: 22px; box-shadow: 0 4px 14px -4px rgba(99,102,241,0.6);
            flex-shrink: 0;
        }
        .iq-header .iq-title { font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; }
        .iq-header .iq-subtitle { color: var(--iq-text-dim); font-size: 0.85rem; margin-top: 1px; }
        .iq-header .iq-spacer { flex: 1; }
        .iq-chip {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--iq-surface-2); border: 1px solid var(--iq-border-strong);
            border-radius: 999px; padding: 5px 12px; font-size: 0.8rem; color: var(--iq-text-dim);
        }

        .iq-badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 3px 10px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
            border: 1px solid currentColor;
        }
        .iq-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

        .iq-stats { display: flex; gap: 12px; margin: 1rem 0; flex-wrap: wrap; }
        .iq-stat {
            flex: 1; min-width: 150px;
            background: var(--iq-surface-2); border: 1px solid var(--iq-border);
            border-radius: var(--iq-radius); padding: 14px 16px;
        }
        .iq-stat-label {
            font-size: 0.72rem; color: var(--iq-text-faint); font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;
        }
        .iq-stat-value { font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em; }
        .iq-stat-sub { font-size: 0.78rem; color: var(--iq-text-dim); margin-top: 2px; }

        .iq-panel {
            background: var(--iq-surface-2); border: 1px solid var(--iq-border);
            border-left: 3px solid var(--iq-primary);
            border-radius: var(--iq-radius); padding: 16px 18px; margin: 0.75rem 0;
            line-height: 1.55; color: var(--iq-text);
        }
        .iq-panel-label {
            font-size: 0.72rem; font-weight: 700; color: var(--iq-primary);
            text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
        }

        .iq-diff-label {
            font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.04em; color: var(--iq-text-faint); margin-bottom: 6px;
        }
        .iq-diff-text {
            background: var(--iq-surface); border: 1px solid var(--iq-border);
            border-radius: var(--iq-radius-sm); padding: 10px 12px;
            font-size: 0.87rem; line-height: 1.5; color: var(--iq-text-dim);
        }

        .iq-empty {
            text-align: center; padding: 3rem 1rem; color: var(--iq-text-faint);
            border: 1px dashed var(--iq-border-strong); border-radius: var(--iq-radius);
        }
        .iq-empty .iq-empty-icon { font-size: 2rem; margin-bottom: 8px; opacity: 0.6; }

        /* auth screen */
        .iq-auth-wrap { max-width: 420px; margin: 3vh auto 0 auto; }
        .iq-auth-brand { text-align: center; margin-bottom: 1.75rem; }
        .iq-auth-brand .iq-mark {
            width: 56px; height: 56px; border-radius: 15px; margin: 0 auto 14px auto;
            background: linear-gradient(145deg, var(--iq-primary), #8b5cf6);
            display: flex; align-items: center; justify-content: center;
            font-size: 28px; box-shadow: 0 8px 24px -6px rgba(99,102,241,0.55);
        }
        .iq-auth-brand h1 { font-size: 1.6rem !important; margin: 0 !important; }
        .iq-auth-brand p { color: var(--iq-text-dim); font-size: 0.9rem; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def severity_badge_html(severity: str | None) -> str:
    c = SEVERITY_COLORS.get(severity or "", _DEFAULT_SEVERITY)
    label = (severity or "n/a").upper()
    return (
        f'<span class="iq-badge" style="color:{c["fg"]}; background:{c["bg"]}; border-color:{c["border"]}55;">'
        f'<span class="iq-dot"></span>{label}</span>'
    )


def stat_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="iq-stat-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="iq-stat"><div class="iq-stat-label">{label}</div>'
        f'<div class="iq-stat-value">{value}</div>{sub_html}</div>'
    )


def header(subtitle: str, right_html: str = "") -> None:
    st.markdown(
        f"""
        <div class="iq-header">
            <div class="iq-mark">🛡️</div>
            <div>
                <div class="iq-title">IntegrityIQ</div>
                <div class="iq-subtitle">{subtitle}</div>
            </div>
            <div class="iq-spacer"></div>
            {right_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, text: str) -> None:
    st.markdown(
        f'<div class="iq-empty"><div class="iq-empty-icon">{icon}</div>{text}</div>',
        unsafe_allow_html=True,
    )
