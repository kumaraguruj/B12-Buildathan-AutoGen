"""Operations dashboard: styled KPI cards + polished Plotly graphs."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import database

# Cohesive color palette
COLORS = ["#1a73e8", "#0d9488", "#8b5cf6", "#f59e0b", "#ef4444", "#ec4899", "#6366f1", "#14b8a6"]
PLOTLY_TEMPLATE = "plotly_white"
STATUS_COLORS = {"Resolved": "#0d9488", "Escalated": "#f59e0b", "In Progress": "#1a73e8"}
SENTIMENT_COLORS = {"Positive": "#0d9488", "Neutral": "#94a3b8", "Negative": "#ef4444"}
PRIORITY_COLORS = {"Low": "#0d9488", "Medium": "#1a73e8", "High": "#f59e0b", "Critical": "#ef4444"}


def _kpi_card(label: str, value: str, style: str = "", delta: str = "", delta_dir: str = "") -> str:
    """Return HTML for one styled KPI card."""
    delta_html = ""
    if delta:
        cls = "up" if delta_dir == "up" else "down"
        arrow = "↑" if delta_dir == "up" else "↓"
        delta_html = f'<p class="kpi-delta {cls}">{arrow} {delta}</p>'
    return f"""
    <div class="kpi-card {style}">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{value}</p>
        {delta_html}
    </div>
    """


def _kpi_section(title: str, cards: list[tuple[str, str, str, str, str]]) -> None:
    st.markdown(f'<p class="section-hdr">{title}</p>', unsafe_allow_html=True)
    for column, card in zip(st.columns(len(cards)), cards):
        column.markdown(_kpi_card(*card), unsafe_allow_html=True)


def render() -> None:
    df, runs = database.load_tickets(), database.load_runs()
    if df.empty:
        st.info("No tickets yet. Start a support chat to generate data.")
        return

    # ---- Filters ----
    st.markdown('<p class="section-hdr">🔍 Filters</p>', unsafe_allow_html=True)
    f1, f2 = st.columns([2, 3])
    days = f1.selectbox("Period", [7, 14, 30, 90], index=2, format_func=lambda d: f"Last {d} days")
    prods = f2.multiselect("Products", sorted(df["product"].unique()), default=[])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    d = df[df["created_at"] >= cutoff]
    if prods:
        d = d[d["product"].isin(prods)]
    prev = df[(df["created_at"] < cutoff) & (df["created_at"] >= cutoff - pd.Timedelta(days=days))]

    # ---- Compute KPIs ----
    total = len(d)
    open_cnt = int(d["status"].isin(["Escalated", "In Progress"]).sum())
    ai_rate = (d["resolved_by"].eq("AI Agents").mean() * 100) if total else 0
    csat = d["csat"].dropna().mean()
    avg_res = d["resolution_minutes"].dropna().mean()
    critical = int(d["priority"].eq("Critical").sum())
    avg_lat = runs[runs["ts"] >= cutoff]["latency_s"].mean()
    photo_cnt = int(d["has_image"].sum())
    neg_pct = (d["sentiment"].eq("Negative").mean() * 100) if total else 0

    prev_total = len(prev)
    delta_total = str(total - prev_total) if prev_total else ""
    delta_dir = "up" if total > prev_total else "down"

    _kpi_section("📊 Performance", [
        ("Total Tickets", str(total), "", delta_total, delta_dir),
        ("Open / Escalated", str(open_cnt), "amber" if open_cnt > 5 else "green", "", ""),
        ("AI Resolution Rate", f"{ai_rate:.0f}%", "green" if ai_rate >= 70 else "amber", "", ""),
        ("Critical Cases", str(critical), "red" if critical > 3 else "green", "", ""),
    ])

    csat_style = "green" if pd.notna(csat) and csat >= 4 else ("amber" if pd.notna(csat) and csat >= 3 else "red")
    _kpi_section("😊 Customer Experience", [
        ("Avg CSAT (1–5)", f"{csat:.2f}" if pd.notna(csat) else "–", csat_style, "", ""),
        ("Negative Sentiment", f"{neg_pct:.0f}%" if total else "–",
         "red" if neg_pct > 25 else "green", "", ""),
        ("Tickets with Photos", str(photo_cnt), "purple", "", ""),
    ])
    _kpi_section("⚡ Operations", [
        ("Avg Resolution (min)", f"{avg_res:.1f}" if pd.notna(avg_res) else "–",
         "green" if pd.notna(avg_res) and avg_res < 60 else "amber", "", ""),
        ("Avg Agent Latency (s)", f"{avg_lat:.1f}" if pd.notna(avg_lat) else "–",
         "green" if pd.notna(avg_lat) and avg_lat < 4 else "amber", "", ""),
    ])

    # ---- Health gauge ----
    health = min(100, max(0, int(ai_rate * 0.4 + (csat / 5 * 100 if pd.notna(csat) else 50) * 0.3
                                  + max(0, 100 - neg_pct * 2) * 0.3)))
    st.markdown('<p class="section-hdr">🩺 Ticket Health Score</p>', unsafe_allow_html=True)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health,
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color="#1a73e8"),
            steps=[
                dict(range=[0, 40], color="#fee2e2"),
                dict(range=[40, 70], color="#fef3c7"),
                dict(range=[70, 100], color="#d1fae5"),
            ],
            threshold=dict(line=dict(color="#ef4444", width=2), thickness=0.8, value=40),
        ),
        number=dict(suffix="/100", font=dict(size=28)),
        title=dict(text="Overall Health", font=dict(size=14)),
    ))
    gauge.update_layout(height=220, margin=dict(l=30, r=30, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(gauge, use_container_width=True)

    st.divider()

    # ---- Graphs ----
    st.markdown('<p class="section-hdr">📈 Trends & Breakdown</p>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    daily = d.set_index("created_at").resample("D").size().reset_index(name="tickets")
    fig1 = px.area(daily, x="created_at", y="tickets", markers=True, title="Tickets per Day",
                   template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLORS[0]])
    fig1.update_traces(fill="tozeroy", fillcolor="rgba(26,115,232,0.1)")
    fig1.update_layout(xaxis_title="", yaxis_title="Tickets", height=320)
    g1.plotly_chart(fig1, use_container_width=True)

    cat_data = d["category"].value_counts().reset_index()
    fig2 = px.bar(cat_data, x="count", y="category", orientation="h", title="Tickets by Category",
                  template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLORS[1]])
    fig2.update_layout(xaxis_title="Count", yaxis_title="", height=320)
    g2.plotly_chart(fig2, use_container_width=True)

    g3, g4, g5 = st.columns(3)
    fig3 = px.pie(d, names="resolved_by", title="Resolved By", hole=0.5,
                  template=PLOTLY_TEMPLATE, color="resolved_by",
                  color_discrete_map={"AI Agents": COLORS[0], "Human Technician": COLORS[3]})
    fig3.update_layout(height=300, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    g3.plotly_chart(fig3, use_container_width=True)

    fig4 = px.pie(d, names="sentiment", title="Customer Sentiment", hole=0.5,
                  template=PLOTLY_TEMPLATE, color="sentiment", color_discrete_map=SENTIMENT_COLORS)
    fig4.update_layout(height=300, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    g4.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(d, x="priority", category_orders={"priority": config.PRIORITIES},
                        title="Priority Mix", template=PLOTLY_TEMPLATE, color="priority",
                        color_discrete_map=PRIORITY_COLORS)
    fig5.update_layout(height=300, showlegend=False, xaxis_title="", yaxis_title="Count")
    g5.plotly_chart(fig5, use_container_width=True)

    # ---- CSAT Trend ----
    csat_trend = (d.dropna(subset=["csat"]).set_index("created_at")
                  .resample("W")["csat"].mean().reset_index())
    if not csat_trend.empty:
        fig_csat = px.line(csat_trend, x="created_at", y="csat", markers=True,
                           title="Weekly CSAT Trend", range_y=[1, 5],
                           template=PLOTLY_TEMPLATE, color_discrete_sequence=[COLORS[0]])
        fig_csat.add_hline(y=4, line_dash="dash", line_color="#0d9488", annotation_text="Target (4.0)")
        fig_csat.update_layout(height=300, xaxis_title="", yaxis_title="CSAT")
        st.plotly_chart(fig_csat, use_container_width=True)

    # ---- Recent tickets table ----
    st.markdown('<p class="section-hdr">🗂️ Recent Tickets</p>', unsafe_allow_html=True)
    recent = d.sort_values("created_at", ascending=False).head(50)
    if not recent.empty:
        display_cols = ["id", "created_at", "customer", "product", "category",
                        "priority", "sentiment", "status", "resolved_by", "csat"]
        show = recent[[c for c in display_cols if c in recent.columns]].copy()
        st.dataframe(
            show.style.map(
                lambda v: "background-color: #d1fae5; color: #065f46" if v == "Resolved"
                else ("background-color: #fef3c7; color: #92400e" if v == "Escalated"
                      else ("background-color: #dbeafe; color: #1e40af" if v == "In Progress" else "")),
                subset=["status"]
            ).map(
                lambda v: "color: #ef4444; font-weight:600" if v == "Critical"
                else ("color: #f59e0b; font-weight:600" if v == "High" else ""),
                subset=["priority"]
            ),
            use_container_width=True, hide_index=True, height=400
        )
