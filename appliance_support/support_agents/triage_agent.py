"""Agent 2 - Triage Agent (AutoGen 0.7.5).

Responsibility: classify the request (product, category, priority, sentiment) and extract identifiers.
"""
from __future__ import annotations

import config
from support_agents.base import ask, extract_json, format_history

NAME = "TriageAgent"
SYSTEM = (
    f"{config.IDENTITY} You are the Triage Agent of the support desk. The customer may write in Tamil, English or "
    "Tamil in English letters. From the customer message, (optional) image analysis and chat history, output ONLY a "
    f"JSON object with keys: product (one of {config.PRODUCTS} or 'Unknown'), category (one of {config.CATEGORIES}), "
    f"priority (one of {config.PRIORITIES}; Critical for gas smell, fire, sparks, electric shock, flooding), "
    f"sentiment (one of {config.SENTIMENTS}), serial_number (string or null), error_code (string or null), "
    "customer_name (string or null), search_query (a concise retrieval query written in ENGLISH, "
    "because the knowledge base is in English). All values must use the English labels above.")


def run(user_text: str, vision: str, history: list[dict], customer: str) -> dict:
    raw = ask(NAME, SYSTEM,
              f"History:\n{format_history(history)}\n\nCustomer message: {user_text}\n"
              f"Image analysis: {vision or 'none'}",
              "Triage", model=config.FAST_MODEL, temperature=0.0)
    return normalize(raw, customer, user_text)


def normalize(raw: str, customer: str, user_text: str) -> dict:
    """Validate the triage agent's JSON and apply safe defaults."""
    d = extract_json(raw)
    category, priority, sentiment = d.get("category"), d.get("priority"), d.get("sentiment")
    product = d.get("product")
    serial = d.get("serial_number")
    error_code = d.get("error_code")
    customer_name = d.get("customer_name")
    search_query = d.get("search_query")
    return {
        "product": product if product in config.PRODUCTS else "Unknown",
        "category": category if category in config.CATEGORIES else "Other",
        "priority": priority if priority in config.PRIORITIES else "Medium",
        "sentiment": sentiment if sentiment in config.SENTIMENTS else "Neutral",
        "serial_number": serial[:80].strip() if isinstance(serial, str) else None,
        "error_code": error_code[:80].strip() if isinstance(error_code, str) else None,
        "customer_name": customer_name[:100].strip() if isinstance(customer_name, str) and customer_name.strip()
        else customer or "Guest",
        "search_query": search_query[:500].strip() if isinstance(search_query, str) and search_query.strip()
        else user_text,
    }
