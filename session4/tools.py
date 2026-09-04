"""Session 4 - tool definitions and implementations for the supply chain agent.

Four tools, deliberately distinct so the agent has to route rather than
always reach for the same one:

- search_knowledge_base: unstructured retrieval over the RAG knowledge base
  (concepts, definitions, best practices).
- calculate_reorder_point: deterministic math, so the model does the routing
  instead of hallucinating arithmetic.
- lookup_shipment_status: a structured "system of record" read, backed by
  SQLite (db.py) rather than a mock dict.
- flag_shipment_for_expedite: the one tool with a side effect. It writes to
  the same SQLite table, gated behind a confirm flag - see the docstring on
  the function for how the guardrail works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session3.retrieval import retrieve
from session4.db import get_connection


def search_knowledge_base(query: str, k: int = 4) -> dict:
    hits = retrieve(query, k=k)
    return {
        "results": [
            {"source": h["source"], "text": h["text"], "similarity": round(h["similarity"], 3)}
            for h in hits
        ]
    }


def calculate_reorder_point(
    avg_daily_demand: float,
    lead_time_days: float,
    safety_stock: float = 0.0,
) -> dict:
    reorder_point = avg_daily_demand * lead_time_days + safety_stock
    return {
        "reorder_point": round(reorder_point, 2),
        "formula": "reorder_point = avg_daily_demand * lead_time_days + safety_stock",
    }


def lookup_shipment_status(shipment_id: str) -> dict:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM shipments WHERE shipment_id = ?", (shipment_id.upper(),)
    ).fetchone()
    conn.close()
    if row is None:
        return {"error": f"No shipment found with ID '{shipment_id}'"}
    return dict(row)


def flag_shipment_for_expedite(shipment_id: str, reason: str, confirm: bool = False) -> dict:
    """Write a side effect - flags a shipment for expedited handling.

    Defaults to a dry run (confirm=False): looks the shipment up, returns a
    preview of what would change, and writes nothing. The agent's system
    prompt requires it to relay that preview to the user and get explicit
    confirmation before calling this again with confirm=True to actually
    apply it. The gate lives here in code, not just in the prompt - even if
    the model ignores its instructions, a first call can never write.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM shipments WHERE shipment_id = ?", (shipment_id.upper(),)
    ).fetchone()
    if row is None:
        conn.close()
        return {"error": f"No shipment found with ID '{shipment_id}'"}

    if not confirm:
        conn.close()
        return {
            "status": "confirmation_required",
            "shipment_id": shipment_id.upper(),
            "preview": (
                f"Would flag {shipment_id.upper()} (carrier {row['carrier']}, "
                f"currently '{row['status']}') for expedite. Reason: {reason}. "
                "This is a dry run - nothing has been written yet."
            ),
        }

    conn.execute(
        "UPDATE shipments SET expedite_requested = 1, expedite_reason = ? WHERE shipment_id = ?",
        (reason, shipment_id.upper()),
    )
    conn.commit()
    conn.close()
    return {
        "status": "flagged",
        "shipment_id": shipment_id.upper(),
        "expedite_reason": reason,
    }


TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the supply chain knowledge base for conceptual or "
            "procedural information (definitions, frameworks, best "
            "practices). Use this for 'what is' / 'how does' / 'why' "
            "questions. Not for specific shipment records or arithmetic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query."},
                "k": {"type": "integer", "description": "Number of chunks to retrieve.", "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_reorder_point",
        "description": (
            "Compute a reorder point given average daily demand, lead time "
            "in days, and optional safety stock. Use this instead of doing "
            "the arithmetic yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "avg_daily_demand": {"type": "number"},
                "lead_time_days": {"type": "number"},
                "safety_stock": {"type": "number", "default": 0},
            },
            "required": ["avg_daily_demand", "lead_time_days"],
        },
    },
    {
        "name": "lookup_shipment_status",
        "description": (
            "Look up the live status, carrier, origin/destination, and ETA "
            "of a specific shipment by its ID (e.g. 'SH-1001'). Use this "
            "for questions about a specific shipment, not general shipping "
            "concepts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "shipment_id": {"type": "string"},
            },
            "required": ["shipment_id"],
        },
    },
    {
        "name": "flag_shipment_for_expedite",
        "description": (
            "Flag a shipment for expedited handling. This WRITES to the "
            "shipment record, unlike the other tools. Always call it first "
            "with confirm=false (the default) - this only previews the "
            "change and writes nothing. Relay the preview to the user in "
            "plain text and wait for their explicit yes/confirmation in a "
            "later message before calling again with confirm=true. Never "
            "set confirm=true unless the user has explicitly agreed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "shipment_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why expedite is being requested."},
                "confirm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Must be true to actually write. False (default) previews only.",
                },
            },
            "required": ["shipment_id", "reason"],
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "search_knowledge_base": lambda **kwargs: search_knowledge_base(**kwargs),
    "calculate_reorder_point": lambda **kwargs: calculate_reorder_point(**kwargs),
    "lookup_shipment_status": lambda **kwargs: lookup_shipment_status(**kwargs),
    "flag_shipment_for_expedite": lambda **kwargs: flag_shipment_for_expedite(**kwargs),
}
