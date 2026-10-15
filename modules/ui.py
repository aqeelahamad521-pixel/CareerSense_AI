"""Shared presentation primitives for the CareerSense workspace."""
from html import escape

import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --cs-ink: #131209;
            --cs-muted: rgba(19, 18, 9, .62);
            --cs-soft: rgba(19, 18, 9, .045);
            --cs-line: rgba(19, 18, 9, .12);
            --cs-surface: #ffffff;
            --cs-radius: .75rem;
        }

        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: "Manrope", system-ui, sans-serif;
            color: var(--cs-ink);
        }

        [data-testid="stAppViewContainer"] { background: var(--cs-surface); }
        [data-testid="stHeader"] { background: rgba(255, 255, 255, .88); }
        [data-testid="stMainBlockContainer"] {
            max-width: 86rem;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        [data-testid="stSidebar"] { background: #131209; }
        [data-testid="stSidebar"] * { color: #ffffff; }
        [data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, .16); }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: rgba(255, 255, 255, .66); }
        [data-testid="stSidebar"] button {
            border-color: rgba(255, 255, 255, .28);
            background: transparent;
        }
        [data-testid="stSidebar"] button:hover { border-color: rgba(255, 255, 255, .72); }

        h1, h2, h3, h4 {
            color: var(--cs-ink);
            letter-spacing: -.035em;
            text-wrap: balance;
        }
        h1 { font-size: clamp(2rem, 4vw, 3.25rem) !important; line-height: 1.05 !important; }
        h2 { font-size: clamp(1.5rem, 2.5vw, 2.1rem) !important; }
        p, li, [data-testid="stCaptionContainer"] { text-wrap: pretty; }

        .cs-eyebrow {
            color: var(--cs-muted);
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .11em;
            margin-bottom: .5rem;
            text-transform: uppercase;
        }
        .cs-page-header { margin: .5rem 0 2rem; max-width: 54rem; }
        .cs-page-header h1 {
            background: linear-gradient(90deg, #000000, #666666);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin: 0 0 .75rem;
        }
        .cs-page-header p { color: var(--cs-muted); font-size: 1rem; margin: 0; line-height: 1.65; }
        .cs-section { margin: 2.5rem 0 1rem; }
        .cs-section h2 { margin: 0 0 .35rem; }
        .cs-section p { color: var(--cs-muted); margin: 0; }
        .cs-empty {
            background: var(--cs-soft);
            border: 1px solid var(--cs-line);
            border-radius: var(--cs-radius);
            padding: 2rem;
            text-align: center;
        }
        .cs-empty strong { display: block; margin-bottom: .5rem; }
        .cs-empty span { color: var(--cs-muted); }
        .cs-brand { font-size: 1.05rem; font-weight: 800; letter-spacing: -.03em; }
        .cs-badge {
            border: 1px solid var(--cs-line);
            border-radius: 999px;
            display: inline-flex;
            font-size: .72rem;
            font-weight: 700;
            padding: .25rem .6rem;
            text-transform: capitalize;
        }

        [data-testid="stMetric"] {
            background: var(--cs-soft);
            border: 1px solid var(--cs-line);
            border-radius: var(--cs-radius);
            padding: 1rem;
        }
        [data-testid="stMetricValue"] {
            color: var(--cs-ink);
            font-size: 1.75rem;
            font-variant-numeric: tabular-nums;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] { color: var(--cs-muted); }

        .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
            border-radius: .625rem;
            font-weight: 650;
            min-height: 2.75rem;
            transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease, transform 100ms ease;
        }
        .stButton > button:active, .stFormSubmitButton > button:active, .stDownloadButton > button:active {
            transform: scale(.96);
        }
        button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
            background: #131209;
            border-color: #131209;
            color: #ffffff;
        }
        button[kind="primary"]:hover { background: #272727; border-color: #272727; }

        [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--cs-line);
            gap: .25rem;
            overflow-x: auto;
            scrollbar-width: none;
        }
        [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        [data-baseweb="tab"] {
            border-radius: .5rem .5rem 0 0;
            flex: 0 0 auto;
            font-weight: 600;
            padding: .75rem 1rem;
            white-space: nowrap;
        }
        [data-baseweb="tab-highlight"] { background: #131209; }
        [data-testid="stExpander"] { border-color: var(--cs-line); border-radius: var(--cs-radius); }
        [data-testid="stDataFrame"] { border: 1px solid var(--cs-line); border-radius: var(--cs-radius); overflow: hidden; }
        hr { border-color: var(--cs-line); margin: 2rem 0; }

        input, textarea, [data-baseweb="select"] > div { border-radius: .625rem !important; }
        input:focus-visible, textarea:focus-visible, button:focus-visible, [tabindex]:focus-visible {
            outline: 2px solid #131209 !important;
            outline-offset: 2px !important;
        }

        @media (max-width: 48rem) {
            [data-testid="stMainBlockContainer"] { padding: 3.5rem 1rem 4rem; }
            input, textarea, [data-baseweb="select"] { font-size: 16px !important; }
            .cs-page-header { margin-bottom: 1.5rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, description: str):
    st.markdown(
        f'<div class="cs-page-header"><div class="cs-eyebrow">{escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str = ""):
    body = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f'<div class="cs-section"><h2>{escape(title)}</h2>{body}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, description: str):
    st.markdown(
        f'<div class="cs-empty"><strong>{escape(title)}</strong><span>{escape(description)}</span></div>',
        unsafe_allow_html=True,
    )


def status_badge(label: str):
    return f'<span class="cs-badge">{escape(label.replace("_", " "))}</span>'
