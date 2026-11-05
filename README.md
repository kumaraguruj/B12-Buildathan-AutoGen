# யாழ் சூப்பர் ஸ்டோர்ஸ் · Yaazh Super Stores
_ஒவ்வொரு வீட்டின் நம்பிக்கை— Everything you want_

AI customer-care desk: AutoGen **0.7.5** agents + LangChain + FAISS + Streamlit.

```
appliance_support/
├── app.py                     # Streamlit entry point (sidebar navigation and wiring)
├── orchestrator.py            # Runs the agent pipeline end-to-end
├── config.py                  # Models, paths, constants
├── database.py                # SQLite: tickets, warranty, agent telemetry
├── build_embeddings.py        # Reads kb_docs/*.pdf -> chunks -> embeddings -> FAISS
├── knowledge_base.py          # PDF reader, chunking, idempotent FAISS sync
├── kb_docs/                   # Existing PDF manuals; add or upload more PDFs here
├── tools.py                   # Tools given to AutoGen agents (KB search, warranty, ticket)
├── runtime.py                 # Shared context for tools (vector store, has_image)
├── support_agents/
│   ├── base.py                # AutoGen 0.7.5 helper: ask() runs an AssistantAgent (async) with optional tools
│   ├── vision_agent.py        # 1. Vision Inspector (LangChain + GPT-4o vision)
│   ├── triage_agent.py        # 2. Triage (category / priority / sentiment / serial)
│   ├── knowledge_agent.py     # 3. Knowledge (RAG tool over FAISS)
│   ├── warranty_agent.py      # 4. Warranty (DB tool)
│   ├── resolution_agent.py    # 5. Resolution (drafts customer reply)
│   ├── qa_agent.py            # 6. QA & Safety (approve / escalate)
│   └── escalation_agent.py    # 7. Escalation (ticket tool)
└── ui/
    ├── chat_tab.py            # Guided support chat, photo upload, and CSAT feedback
    ├── dashboard_tab.py       # KPI cards, health score, trends, and ticket table
    └── kb_tab.py              # Upload / test PDF knowledge-base documents
```

## Knowledge base: PDFs -> embeddings
```bash
python build_embeddings.py                 # reads PDFs in kb_docs/, chunks, embeds, stores in FAISS
python build_embeddings.py --query "E21"   # optional retrieval test
python build_embeddings.py --rebuild       # wipe and re-embed everything
```
The app reads existing PDFs in `kb_docs/`; it does not generate PDFs. Add PDFs to that folder or upload them
from the admin Knowledge Base tab. Syncing is incremental: unchanged PDFs are skipped, edited PDFs are
re-embedded, and chunks for deleted PDFs are removed.

## Run
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

## Sign-in
The app has two built-in demo accounts:

| Username | Password | Access |
|---|---|---|
| `user` | `user123` | Support chat |
| `admin` | `admin123` | Support chat, Dashboard, and Knowledge Base |

These in-code credentials are for local/demo use only; replace them with a proper
identity provider or secret-backed authentication before deploying the app publicly.
Admins can switch between Support Chat, Dashboard, and Knowledge Base from the sidebar.
The signed-in header uses the same branded banner as the login screen. The sidebar shows the agent
pipeline after the account and Log out controls.

## Support chat
The guided form lets customers select a request type:

- **Report an appliance issue** — select the product and issue type, then describe the symptoms in chat.
- **Check ticket status** — enter a ticket ID to view its current status and ticket details.
- **Check warranty** — enter the appliance serial number to check coverage and expiry.

Customers can also continue using free-text chat and attach a supported photo. Dashboard reporting
includes KPI cards, a ticket health score, trends, and a recent-ticket table.

Environment overrides: `OPENAI_MODEL`, `OPENAI_FAST_MODEL`, `OPENAI_EMBED_MODEL`, `SUPPORT_DB`, `FAISS_DIR`.

## Adding an agent
1. Create `support_agents/<name>_agent.py` with `NAME`, `SYSTEM` and `run()` (call `ask(..., tools=[...])` from `base.py`).
2. Register it in `support_agents/__init__.py`.
3. Call it from `orchestrator.handle_request`.

## Branding
Store name and tagline live in `config.py` (`STORE_NAME_TA/EN`, `TAGLINE_TA/EN`). They feed the UI and the agent
prompts. Customer replies are in English by default; an explicit language request changes the response language
for the conversation until the customer asks to switch again.

## AutoGen 0.7.5 notes
Agents are `autogen_agentchat.agents.AssistantAgent` objects with an `OpenAIChatCompletionClient`; tools are plain
Python functions. The text workflow runs in a fixed-order `RoundRobinGroupChat` and ends when the escalation agent
emits `PIPELINE_COMPLETE`, with a six-turn cap. Image analysis runs before the team when a photo is attached.
Streamlit is synchronous, so the team runs inside `asyncio.run()` (see `orchestrator.py`).

## Guardrails and observability
Support requests are limited to 4,000 characters, six recent history messages, and 10 MB PNG/JPEG/WebP images.
Warranty lookups are limited to serial numbers present in the customer's message or image analysis. QA output
is validated; malformed decisions and High/Critical priorities are escalated, and ticket creation happens only
after that validation. Team runs have a 180-second timeout.

The app writes structured JSON events for request and agent starts, completions, failures, and guardrail triggers.
Events include a request reference and timings, but do not include customer messages, names, serial numbers, or
image contents.

PDFs with numbered `Qn.` headings are split into one knowledge-base entry per question; other PDFs are chunked
by page. PDF bookmarks are used as section titles when available.
