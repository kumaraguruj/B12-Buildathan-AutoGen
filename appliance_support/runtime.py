"""Process-wide runtime context shared with agent tools (keeps tools free of Streamlit imports)."""
from __future__ import annotations


class Runtime:
    vectorstore = None            # langchain FAISS instance, set by app.py
    has_image: bool = False       # whether the current request included a photo
    last_ticket_id: str | None = None   # set by the create_service_ticket tool


RUNTIME = Runtime()
