"""Python functions exposed to AutoGen agents as callable tools."""
from __future__ import annotations

import datetime as dt
from typing import Annotated

import database
from runtime import RUNTIME


def search_knowledge_base(query: Annotated[str, "Customer problem, product and error code to look up"]) -> str:
    """Semantic search over appliance manuals, FAQs and policies (vector DB)."""
    vs = RUNTIME.vectorstore
    if vs is None:
        return "Knowledge base is not initialised."
    hits = vs.similarity_search(query, k=4)
    if not hits:
        return "No relevant knowledge-base articles found."
    return "\n\n".join(f"[{d.metadata.get('title')}]\n{d.page_content}" for d in hits)


def check_warranty(serial_number: Annotated[str, "Product serial number, e.g. WM-1001"]) -> str:
    """Check warranty status for a serial number."""
    row = database.lookup_warranty(serial_number)
    if not row:
        return (f"Serial {serial_number} not found. Ask the customer to re-check the label "
                "or share proof of purchase.")
    product, bought, months, customer = row
    expiry = dt.date.fromisoformat(bought) + dt.timedelta(days=30 * months)
    state = "IN WARRANTY" if expiry >= dt.date.today() else "OUT OF WARRANTY"
    return (f"{product} owned by {customer}: purchased {bought}, warranty {months} months, "
            f"expires {expiry} -> {state}.")


def create_service_ticket(
    customer: Annotated[str, "Customer name"],
    product: Annotated[str, "Appliance type"],
    category: Annotated[str, "Issue category"],
    priority: Annotated[str, "Low | Medium | High | Critical"],
    sentiment: Annotated[str, "Positive | Neutral | Negative"],
    summary: Annotated[str, "Short technician-ready summary of the problem and steps already tried"],
) -> str:
    """Open an escalation ticket for a human technician."""
    tid = database.save_ticket(customer, product, category, priority, sentiment, "Escalated",
                               "Human Technician", summary, None, RUNTIME.has_image)
    RUNTIME.last_ticket_id = tid
    return f"Ticket {tid} created and assigned to the technician queue."
