"""Session 4 - tool definitions and implementations for the supply chain agent.

Three tools, deliberately distinct so the agent has to route rather than
always reach for the same one:

- search_knowledge_base: unstructured retrieval over the RAG knowledge base
  (concepts, definitions, best practices).
- calculate_reorder_point: deterministic math, so the model does the routing
  instead of hallucinating arithmetic.
- lookup_shipment_status: a structured "system of record" lookup (mocked
  here as an in-memory dict, standing in for a TMS/ERP call).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session3.retrieval import retrieve

# Mock structured data a real deployment would pull from a TMS/ERP.
_SHIPMENTS = {
    "SH-1001": {
        "carrier": "Meridian Freight",
        "status": "in_transit",
        "origin": "Chicago, IL",
        "destination": "Dallas, TX",
        "eta": "2026-07-11",
    },
    "SH-1002": {
        "carrier": "Coldline Logistics",
        "status": "delayed",
        "origin": "Memphis, TN",
        "destination": "Atlanta, GA",
        "eta": "2026-07-13",
    },
    "SH-1003": {
        "carrier": "Meridian Freight",
        "status": "delivered",
        "origin": "Kansas City, MO",
        "destination": "Omaha, NE",
        "eta": "2026-07-05",
    },
}


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
    record = _SHIPMENTS.get(shipment_id.upper())
    if record is None:
        return {"error": f"No shipment found with ID '{shipment_id}'"}
    return {"shipment_id": shipment_id.upper(), **record}


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
]

TOOL_IMPLEMENTATIONS = {
    "search_knowledge_base": lambda **kwargs: search_knowledge_base(**kwargs),
    "calculate_reorder_point": lambda **kwargs: calculate_reorder_point(**kwargs),
    "lookup_shipment_status": lambda **kwargs: lookup_shipment_status(**kwargs),
}
