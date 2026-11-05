"""Agent 1 - Vision Inspector (LangChain + OpenAI vision model).

Responsibility: turn a customer photo into factual text (model, error code, visible damage).
"""
from __future__ import annotations

import base64
import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import config
from database import log_run

PROMPT = (
    "You are the Vision Inspector for an appliance support desk. Describe the appliance in the photo: brand/"
    "model if visible, any error code or display text, visible damage (rust, leak, burn marks, cracks, ice "
    "build-up), and anything relevant to the customer's complaint. Be factual and brief; say 'not visible' "
    "when unsure. Customer note: {note}")


def run(image_bytes: bytes, mime: str, user_text: str) -> str:
    t0 = time.time()
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    b64 = base64.b64encode(image_bytes).decode()
    msg = HumanMessage(content=[
        {"type": "text", "text": PROMPT.format(note=user_text or "(none)")},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    out = llm.invoke([msg]).content
    log_run("Vision", time.time() - t0)
    return out if isinstance(out, str) else str(out)
