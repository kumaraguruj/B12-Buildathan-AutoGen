"""Agent 3 - Knowledge Agent (AutoGen 0.7.5; RAG over the FAISS vector DB via a tool)."""
from __future__ import annotations

import config
from support_agents.base import ask
from tools import search_knowledge_base

NAME = "KnowledgeAgent"
SYSTEM = (
    f"{config.IDENTITY} You are the Knowledge Agent. ALWAYS call search_knowledge_base with a precise ENGLISH query "
    "(the manuals are in English), then return a concise English bullet summary of the relevant facts and procedures "
    "found (quote error codes and limits exactly). Do not invent information. If nothing relevant is found, say so.")


def run(user_text: str, triage: dict, vision: str) -> str:
    prompt = (f"Customer issue: {user_text}\nProduct: {triage['product']}\n"
              f"Error code: {triage['error_code']}\nImage findings: {vision or 'none'}\n"
              f"Suggested query: {triage['search_query']}")
    return ask(NAME, SYSTEM, prompt, "Knowledge", model=config.CHAT_MODEL, tools=[search_knowledge_base])
