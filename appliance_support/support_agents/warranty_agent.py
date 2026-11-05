"""Agent 4 - Warranty Agent (AutoGen 0.7.5; serial-number lookup via a tool)."""
from __future__ import annotations

import config
from support_agents.base import ask
from tools import check_warranty

NAME = "WarrantyAgent"
SYSTEM = (
    f"{config.IDENTITY} You are the Warranty Agent. If a serial number is provided, call check_warranty and report "
    "the result in one or two English sentences.")

NO_SERIAL = "No serial number provided."


def should_run(triage: dict) -> bool:
    return bool(triage.get("serial_number")) or triage.get("category") == "Warranty / Repair"


def run(user_text: str, triage: dict) -> str:
    if not should_run(triage):
        return NO_SERIAL
    if not triage.get("serial_number"):
        return NO_SERIAL + " Ask the customer for the serial number on the appliance label."
    return ask(NAME, SYSTEM,
               f"Serial number: {triage['serial_number']}. Customer message: {user_text}",
               "Warranty", model=config.FAST_MODEL, temperature=0.0, tools=[check_warranty])
