"""Streamlit entry point.   Run:  streamlit run app.py"""
from __future__ import annotations

import hmac
import os

import streamlit as st

import config
import database
import knowledge_base
from runtime import RUNTIME
from ui import chat_tab, dashboard_tab, kb_tab

ACCOUNTS = {
    "user": ("user123", "user"),
    "admin": ("admin123", "admin"),
}

# ---------------------------------------------------------------------------
# Custom CSS theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- Global ---- */
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stTextInput label { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stDivider { border-color: #334155 !important; }
[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #2563eb !important;
    border: 1px solid #3b82f6 !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color: #60a5fa !important;
}
[data-testid="stSidebar"] button[kind="primary"] * { color: #ffffff !important; }

/* ---- Tabs ---- */
button[data-baseweb="tab"] {
    font-weight: 600; font-size: 1rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1a73e8 !important; border-bottom-color: #1a73e8 !important;
}

/* ---- KPI metric cards ---- */
.kpi-card {
    background: #ffffff; border-radius: 12px; padding: 1.1rem 1.3rem;
    border-left: 4px solid #1a73e8; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 0.5rem;
}
.kpi-card.green  { border-left-color: #0d9488; }
.kpi-card.amber  { border-left-color: #f59e0b; }
.kpi-card.red    { border-left-color: #ef4444; }
.kpi-card.purple { border-left-color: #8b5cf6; }
.kpi-label { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin: 0; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #0f172a; margin: 0.15rem 0 0 0; line-height: 1.1; }
.kpi-delta { font-size: 0.8rem; margin-top: 0.2rem; }
.kpi-delta.up   { color: #0d9488; }
.kpi-delta.down { color: #ef4444; }

/* ---- Section headers ---- */
.section-hdr { font-size: 0.85rem; font-weight: 700; color: #475569; text-transform: uppercase;
               letter-spacing: 0.06em; margin: 1.5rem 0 0.8rem 0; display: flex; align-items: center; gap: 0.4rem; }

/* ---- Status badges ---- */
.badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 9999px;
         font-size: 0.72rem; font-weight: 600; }
.badge-green  { background: #d1fae5; color: #065f46; }
.badge-amber  { background: #fef3c7; color: #92400e; }
.badge-red    { background: #fee2e2; color: #991b1b; }
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-gray   { background: #f1f5f9; color: #475569; }

/* ---- Search result cards ---- */
.result-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.7rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.result-card .title { font-weight: 600; color: #1e293b; margin-bottom: 0.3rem; }
.result-card .meta  { font-size: 0.78rem; color: #64748b; margin-bottom: 0.5rem; }

/* ---- Chat ---- */
.welcome-banner {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border: 1px solid #bfdbfe; border-radius: 14px; padding: 1.5rem;
    margin-bottom: 1rem;
}
.welcome-banner h3 { margin: 0 0 0.5rem 0; color: #1e3a5f; }
.welcome-banner p  { margin: 0.25rem 0; color: #334155; }

/* ---- Sidebar brand ---- */
.sidebar-brand { text-align: center; padding: 0.5rem 0 0.8rem 0; }
.sidebar-brand .logo { font-size: 2.5rem; }
.sidebar-brand .name { font-size: 1.15rem; font-weight: 700; color: #f8fafc !important; }
.sidebar-brand .tagline { font-size: 0.82rem; color: #94a3b8 !important; font-style: italic; }

/* ---- Login page ---- */
.login-banner {
    background: linear-gradient(120deg, #0f172a 0%, #1e3a5f 58%, #0f766e 100%);
    border-radius: 16px;
    color: #ffffff;
    padding: 1.5rem 2rem;
    margin: 0.5rem 0 2.5rem;
}
.login-banner h1 { color: #ffffff; margin: 0; font-size: 2rem; }
.login-banner p { color: #dbeafe; margin: 0.45rem 0 0; }
.login-card-title { color: #0f172a; font-size: 1.5rem; font-weight: 700; }

/* ---- Agent pipeline pills ---- */
.agent-pill { display: inline-block; padding: 0.2rem 0.55rem; border-radius: 8px;
              font-size: 0.7rem; font-weight: 600; margin: 0.15rem 0.1rem;
              background: #1e293b; color: #e2e8f0; }
</style>
"""


def _authenticate(username: str, password: str) -> str | None:
    account = ACCOUNTS.get(username)
    if account and hmac.compare_digest(password, account[0]):
        return account[1]
    return None


def _render_login() -> None:
    st.markdown(f"""
    <div class="login-banner">
        <h1>🛒 {config.STORE_NAME_TA} · {config.STORE_NAME_EN}</h1>
        <p>{config.TAGLINE_EN} · AI-Powered Customer Care</p>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.1, 1])
    with center:
        with st.container(border=True):
            st.markdown('<div class="login-card-title">Welcome </div>', unsafe_allow_html=True)
            st.caption("Sign in to continue to Customer Care")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

            if submitted:
                role = _authenticate(username.strip(), password)
                if role:
                    st.session_state["username"] = username.strip()
                    st.session_state["role"] = role
                    st.rerun()
                st.error("Incorrect username or password.")


def main() -> None:
    st.set_page_config(page_title=f"{config.STORE_NAME_EN} | Customer Care",
                       page_icon="🛒", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if st.session_state.get("role") not in {"user", "admin"}:
        _render_login()
        return

    # ---- Header ----
    st.markdown(f"""
    <div class="login-banner">
        <h1>🛒 {config.STORE_NAME_TA} · {config.STORE_NAME_EN}</h1>
        <p>{config.TAGLINE_EN} · AI-Powered Customer Care</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-brand">
            <div class="logo">🛒</div>
            <div class="name">{config.STORE_NAME_TA}</div>
            <div class="name" style="font-size:0.9rem;">{config.STORE_NAME_EN}</div>
            <div class="tagline">{config.TAGLINE_TA}</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        if st.session_state["role"] == "admin":
            page = st.radio(
                "Navigation",
                ["💬 Support Chat", "📊 Dashboard", "📚 Knowledge Base"],
                label_visibility="collapsed",
                key="admin_navigation",
            )
            st.divider()
        st.caption("💡 Try serials: WM-1001 · RF-2002 · AC-3003 · DW-4004")
        st.divider()
        st.write(f"Signed in as **{st.session_state['username']}**")
        if st.button("Log out", use_container_width=True, type="primary"):
            st.session_state.pop("username", None)
            st.session_state.pop("role", None)
            st.rerun()
        st.divider()
        st.markdown("**🤖 Agent Pipeline**")
        agents = ["Vision", "Triage", "Knowledge", "Warranty", "Resolution", "QA/Safety", "Escalation"]
        pills_html = " ".join(f'<span class="agent-pill">{a}</span>' for a in agents)
        st.markdown(pills_html, unsafe_allow_html=True)

    database.init_db()
    database.seed_demo_data()

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("🔑 Configure the OPENAI_API_KEY environment variable to start.")
        st.stop()

    if "vs" not in st.session_state:
        with st.spinner("Loading knowledge base…"):
            st.session_state["vs"] = knowledge_base.get_vectorstore()
    RUNTIME.vectorstore = st.session_state["vs"]

    if st.session_state["role"] != "admin":
        st.subheader("💬 Support Chat / உதவி")
        chat_tab.render()
    elif page == "📊 Dashboard":
        dashboard_tab.render()
    elif page == "📚 Knowledge Base":
        kb_tab.render()
    else:
        chat_tab.render()


main()
