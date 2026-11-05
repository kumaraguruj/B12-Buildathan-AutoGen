"""Agent 5 - Resolution Agent (AutoGen 0.7.5): drafts the customer-facing answer."""
from __future__ import annotations

import json

import config
from support_agents.base import ask

NAME = "ResolutionAgent"
SYSTEM = (
    f"{config.IDENTITY} You are the Resolution Agent. {config.LANGUAGE_RULE} Write the final customer reply using "
    "ONLY the supplied knowledge, image findings and warranty status (these are in English; translate faithfully). "
    "Be warm, concise and give numbered, safe, step-by-step instructions. Mention warranty status if known. If the "
    "issue needs a technician, say so and explain that a ticket will be raised. Never advise opening sealed "
    f"electrical parts or gas lines. Sign off with the store name and tagline: "
    f"'{config.STORE_NAME_EN} - {config.TAGLINE_EN}'.")


def run(user_text: str, triage: dict, vision: str, knowledge: str, warranty: str, recent: str) -> str:
    return ask(NAME, SYSTEM,
               f"Customer message: {user_text}\nTriage: {json.dumps(triage)}\n"
               f"Image findings: {vision or 'none'}\nKnowledge:\n{knowledge}\nWarranty: {warranty}\n"
               f"Recent chat:\n{recent}", "Resolution", model=config.CHAT_MODEL)
