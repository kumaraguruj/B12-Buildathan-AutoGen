"""Shared AutoGen 0.7.5 helpers used by every agent module.

AutoGen 0.7 (autogen-agentchat) is async and uses model clients instead of llm_config.
Streamlit scripts are synchronous, so `ask()` wraps each agent run in asyncio.run().
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Callable, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from database import log_run


def default_client_factory(model: str, temperature: float) -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(model=model, api_key=os.environ["OPENAI_API_KEY"],
                                      temperature=temperature)


# Tests can replace this with a factory returning autogen_ext.models.replay.ReplayChatCompletionClient.
CLIENT_FACTORY: Callable[[str, float], object] = default_client_factory


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def format_history(history: list[dict], n: int = 6) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history[-n:])


async def _run_agent(name: str, system_message: str, prompt: str, model: str, temperature: float,
                     tools: Sequence[Callable] | None) -> str:
    client = CLIENT_FACTORY(model, temperature)
    try:
        agent = AssistantAgent(
            name,
            model_client=client,
            system_message=system_message,
            tools=list(tools) if tools else None,
            reflect_on_tool_use=bool(tools),   # model writes a final answer after tool results
            max_tool_iterations=3,
        )
        result = await agent.run(task=prompt)
    finally:
        await client.close()
    for msg in reversed(result.messages):
        if getattr(msg, "source", None) == name and isinstance(getattr(msg, "content", None), str):
            return msg.content
    return ""


def ask(name: str, system_message: str, prompt: str, label: str, *, model: str,
        temperature: float = 0.2, tools: Sequence[Callable] | None = None) -> str:
    """Run one AutoGen AssistantAgent to completion (optionally with tools) and log its latency."""
    t0 = time.time()
    text = asyncio.run(_run_agent(name, system_message, prompt, model, temperature, tools))
    log_run(label, time.time() - t0)
    return text.strip()


__all__ = ["AssistantAgent", "ask", "extract_json", "format_history", "CLIENT_FACTORY"]
