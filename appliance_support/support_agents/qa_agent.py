"""Agent 6 - QA & Safety Agent (AutoGen 0.7.5): reviews the draft and decides whether to escalate."""
from __future__ import annotations

import config
from support_agents.base import ask, extract_json

NAME = "QASafetyAgent"
SYSTEM = (
    f"{config.IDENTITY} You are the QA & Safety Agent. {config.LANGUAGE_RULE} Review the drafted reply against the "
    "customer's issue. Output ONLY JSON: "
    '{"approved_reply": str (improved reply, in the requested response language, keeping the store sign-off), '
    '"escalate": bool (true if safety risk, repeated failure, hardware fault, angry customer, '
    'priority Critical/High, or the KB lacks an answer), "reason": str (in English)}.')


def run(user_text: str, triage: dict, draft: str) -> dict:
    raw = ask(NAME, SYSTEM,
              f"Customer message: {user_text}\nPriority: {triage['priority']}\n"
              f"Sentiment: {triage['sentiment']}\nDraft reply:\n{draft}",
              "QA", model=config.FAST_MODEL, temperature=0.0)
    return parse(raw, triage, draft)


def parse(raw: str, triage: dict, draft: str) -> dict:
    """Validate the QA decision and fail closed when it is malformed."""
    d = extract_json(raw)
    reply = d.get("approved_reply")
    decision = d.get("escalate")
    reason = d.get("reason")
    valid = (
        isinstance(reply, str) and bool(reply.strip())
        and isinstance(decision, bool)
        and isinstance(reason, str)
    )
    forced = triage["priority"] in {"High", "Critical"}
    escalate = not valid or forced or decision is True
    if not valid:
        reason = "QA approval was invalid; escalated for technician review."
        reply = ("This issue needs a technician's review. Please stop using the appliance if it "
                 "appears unsafe; we will arrange follow-up.")
    elif forced and not reason:
        reason = f"{triage['priority']} priority requires technician review."
    return {
        "approved_reply": reply.strip(),
        "escalate": escalate,
        "reason": reason,
        "guardrail_triggered": not valid or forced,
    }
