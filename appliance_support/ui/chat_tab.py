"""Support chat tab: text + photo in, multi-agent answer out."""
from __future__ import annotations

import datetime as dt
from html import escape

import streamlit as st

import config
import database
import orchestrator


def _progress_markup(progress: int, label: str, color: str) -> str:
    safe_label = escape(label)
    return f"""
    <div style="margin:0.5rem 0 1rem;">
        <div style="display:flex;justify-content:space-between;gap:1rem;
                    margin-bottom:0.4rem;font-size:0.85rem;color:#334155;">
            <span>{safe_label}</span><strong>{progress}%</strong>
        </div>
        <div role="progressbar" aria-label="{safe_label}" aria-valuenow="{progress}"
             aria-valuemin="0" aria-valuemax="100"
             style="height:10px;background:#e2e8f0;border-radius:999px;overflow:hidden;">
            <div style="height:100%;width:{progress}%;background:{color};
                        border-radius:999px;transition:width 250ms ease;"></div>
        </div>
    </div>
    """


def render() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("response_language", "English")
    st.session_state.setdefault("guided_context", "")

    # ---- Welcome banner ----
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="welcome-banner">
            <h3>🛒 Welcome to {config.STORE_NAME_EN}</h3>
            <p><em>{config.TAGLINE_EN}</em></p>
            <p style="margin-top:0.8rem; font-size:0.88rem; color:#475569;">
                🔧 Describe your appliance issue below — our <strong>7-agent AI team</strong> will diagnose,
                search manuals, check warranty and draft a solution in seconds.<br>
                📷 You can attach a photo of the error display or appliance for visual diagnosis.<br>
                🗣️ Replies are in English by default. Ask for another language at any time.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---- Guided request ----
    with st.container(border=True):
        st.markdown("#### What would you like help with?")
        col_type, col_product = st.columns(2)
        with col_type:
            request_type = st.selectbox(
                "I want to",
                ["Report an appliance issue", "Check ticket status", "Check warranty"],
                key="guided_request_type",
            )
        with col_product:
            product = st.selectbox("Product", config.PRODUCTS, key="guided_product")

        if request_type == "Report an appliance issue":
            issue_type = st.selectbox("Issue type", config.CATEGORIES, key="guided_issue_type")
            if st.button("Continue to chat", type="primary", key="guided_continue"):
                st.session_state["guided_context"] = (
                    f"Product: {product}. Issue type: {issue_type}."
                )
                st.rerun()
            if st.session_state["guided_context"]:
                st.caption(
                    f"Selected: {product} · {issue_type}. Describe the symptoms in the chat below."
                )
        elif request_type == "Check ticket status":
            ticket_id = st.text_input("Ticket ID", placeholder="e.g. TKT-1234ABCD",
                                      key="guided_ticket_id")
            if st.button("Check ticket", type="primary", key="guided_check_ticket"):
                if not ticket_id.strip():
                    st.warning("Enter your ticket ID to check its status.")
                else:
                    ticket = database.lookup_ticket(ticket_id)
                    user_message = f"Please check the status of ticket {ticket_id.strip().upper()}."
                    if ticket:
                        _, ticket_status, ticket_product, category, created_at = ticket
                        assistant_message = (
                            f"**Ticket {ticket[0]}** is **{ticket_status}**.\n\n"
                            f"Product: {ticket_product} · Issue: {category} · Created: {created_at}"
                        )
                    else:
                        assistant_message = (
                            "I couldn't find that ticket. Please verify the ticket ID and try again."
                        )
                    st.session_state.messages.extend([
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message},
                    ])
                    st.rerun()
        else:
            serial = st.text_input("Appliance serial number", placeholder="e.g. WM-1001",
                                   key="guided_serial")
            if st.button("Check warranty", type="primary", key="guided_check_warranty"):
                if not serial.strip():
                    st.warning("Enter the serial number shown on your appliance.")
                else:
                    warranty = database.lookup_warranty(serial)
                    user_message = f"Please check the warranty for serial {serial.strip().upper()}."
                    if warranty:
                        warranty_product, purchase_date, months, _ = warranty
                        expiry = dt.date.fromisoformat(purchase_date) + dt.timedelta(days=30 * months)
                        coverage = "In warranty" if expiry >= dt.date.today() else "Out of warranty"
                        assistant_message = (
                            f"**{coverage}**\n\nProduct: {warranty_product} · Purchased: {purchase_date} · "
                            f"Warranty: {months} months · Expires: {expiry.isoformat()}"
                        )
                    else:
                        assistant_message = (
                            "I couldn't find that serial number. Please check the appliance label or share "
                            "your proof of purchase with support."
                        )
                    st.session_state.messages.extend([
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message},
                    ])
                    st.rerun()

    # ---- Inputs ----
    col_name, col_upload = st.columns([1, 2])
    with col_name:
        cust = st.text_input("👤 Your name (optional)", key="cust_name", placeholder="e.g. Arun")
    with col_upload:
        up = st.file_uploader("📷 Attach a photo (optional)", type=["png", "jpg", "jpeg", "webp"],
                              max_upload_size=10, key="img")

    # ---- Chat history ----
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("image"):
                st.image(m["image"], width=240)

    # ---- Chat input ----
    prompt = st.chat_input("Describe your appliance problem…")
    if prompt:
        guided_context = st.session_state.pop("guided_context", "")
        if guided_context:
            prompt = f"{guided_context}\nCustomer details: {prompt}"
        image = (up.getvalue(), up.type) if up else None
        history = list(st.session_state.messages)
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "image": up.getvalue() if up else None})
        with st.chat_message("user"):
            st.markdown(prompt)
            if up:
                st.image(up.getvalue(), width=240)

        with st.chat_message("assistant"):
            with st.status("🤖 Agents working…", expanded=True) as status:
                progress = st.empty()
                progress.markdown(_progress_markup(0, "Starting support agents…", "#2563eb"),
                                  unsafe_allow_html=True)
                stages = {
                    "Vision Inspector": 1,
                    "Triage Agent": 2,
                    "Knowledge Agent": 3,
                    "Warranty Agent": 4,
                    "Resolution Agent": 5,
                    "QA & Safety Agent": 6,
                    "Escalation Agent": 7,
                }

                def _cb(s: str) -> None:
                    status.write(s)
                    stage = next((number for agent, number in stages.items() if agent in s), 0)
                    percent = round(stage / 7 * 100)
                    progress.markdown(
                        _progress_markup(percent, f"Step {stage}/7 · {s}", "#2563eb"),
                        unsafe_allow_html=True,
                    )

                try:
                    result = orchestrator.handle_request(
                        prompt, image, history, cust, status_cb=_cb,
                        response_language=st.session_state["response_language"],
                    )
                except ValueError as exc:
                    status.update(label="⚠️ Request needs attention", state="error", expanded=True)
                    st.error(str(exc))
                    return
                except orchestrator.SupportRequestError as exc:
                    status.update(label="❌ Request failed", state="error", expanded=True)
                    st.error(f"{exc} Please share this reference with support.")
                    return
                completion_color = "#16a34a" if result.status == "Resolved" else "#d97706"
                st.session_state["response_language"] = result.response_language
                progress.markdown(
                    _progress_markup(
                        100, f"{result.status} · Agent workflow complete", completion_color
                    ),
                    unsafe_allow_html=True,
                )
                status.update(label=f"✅ Done in {result.seconds}s", state="complete", expanded=False)
            st.markdown(result.reply)

            # ---- Rich agent trace ----
            with st.expander("🔍 Agent trace & diagnostics"):
                trace_tabs = st.tabs(["🧭 Triage", "📚 Knowledge", "🛡️ Warranty", "✅ QA", "🖼️ Vision"])
                with trace_tabs[0]:
                    if result.triage:
                        t = result.triage
                        tcols = st.columns(4)
                        tcols[0].markdown(f"**Product:** {t.get('product', '–')}")
                        tcols[1].markdown(f"**Category:** {t.get('category', '–')}")
                        tcols[2].markdown(_priority_badge(t.get('priority', '–')))
                        tcols[3].markdown(_sentiment_badge(t.get('sentiment', '–')))
                        if t.get("error_code"):
                            st.code(f"Error code: {t['error_code']}")
                        if t.get("search_query"):
                            st.caption(f"🔎 Search query: {t['search_query']}")
                    else:
                        st.caption("No triage data.")
                with trace_tabs[1]:
                    if result.knowledge:
                        st.markdown(result.knowledge)
                    else:
                        st.caption("No knowledge-base results.")
                with trace_tabs[2]:
                    if result.warranty:
                        st.info(result.warranty)
                    else:
                        st.caption("Warranty check not applicable.")
                with trace_tabs[3]:
                    if result.qa:
                        qa = result.qa
                        if qa.get("escalate"):
                            st.warning(f"⚠️ Escalated — {qa.get('reason', '')}")
                        else:
                            st.success("✅ Approved by QA")
                        if qa.get("reason"):
                            st.caption(f"Reason: {qa['reason']}")
                    else:
                        st.caption("No QA data.")
                with trace_tabs[4]:
                    if result.vision:
                        st.markdown(result.vision)
                    else:
                        st.caption("No image was analysed.")

        st.session_state.messages.append({"role": "assistant", "content": result.reply})
        st.session_state["last_ticket"] = result.ticket_id

    # ---- Feedback ----
    ticket = st.session_state.get("last_ticket")
    if ticket:
        st.divider()
        st.markdown("**How was your experience?**")
        score = st.feedback("stars", key=f"fb_{ticket}")
        if score is not None:
            database.set_csat(ticket, score + 1)
            st.success("🙏 Thanks for your feedback!")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _priority_badge(p: str) -> str:
    colors = {"Critical": "red", "High": "amber", "Medium": "blue", "Low": "green"}
    cls = colors.get(p, "gray")
    return f'**Priority:** <span class="badge badge-{cls}">{p}</span>'


def _sentiment_badge(s: str) -> str:
    colors = {"Positive": "green", "Neutral": "gray", "Negative": "red"}
    cls = colors.get(s, "gray")
    return f'**Sentiment:** <span class="badge badge-{cls}">{s}</span>'
