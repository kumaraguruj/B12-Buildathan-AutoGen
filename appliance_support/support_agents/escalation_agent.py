"""Agent 7 - Escalation Agent (AutoGen 0.7.5): opens a technician ticket through a tool."""
from __future__ import annotations

import config
from runtime import RUNTIME
from support_agents.base import ask
from tools import create_service_ticket

NAME = "EscalationAgent"
SYSTEM = (
    f"{config.IDENTITY} You are the Escalation Agent. Read the QA decision and triage data in the team history. "
    "Do not create tickets or claim that one has already been created. The application creates tickets only "
    "after validating the QA decision. Always finish your response with the exact marker PIPELINE_COMPLETE.")


def run(triage: dict, summary: str) -> str | None:
    RUNTIME.last_ticket_id = None
    ask(NAME, SYSTEM,
        f"Customer: {triage['customer_name']}\nProduct: {triage['product']}\n"
        f"Category: {triage['category']}\nPriority: {triage['priority']}\n"
        f"Sentiment: {triage['sentiment']}\nSummary: {summary}",
        "Escalation", model=config.FAST_MODEL, temperature=0.0, tools=[create_service_ticket])
    # The tool records the id it created, which is more reliable than parsing the model's text.
    return RUNTIME.last_ticket_id
