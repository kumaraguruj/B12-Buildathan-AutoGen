"""One module per agent (AutoGen 0.7.5). Each exposes NAME, SYSTEM and run()."""
from support_agents import (  # noqa: F401
    escalation_agent,
    knowledge_agent,
    qa_agent,
    resolution_agent,
    triage_agent,
    vision_agent,
    warranty_agent,
)
