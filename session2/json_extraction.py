"""Session 2 - structured JSON extraction via a forced tool call.

Same pattern reused (and generalized) in session4's agentic tool loop:
give the model a JSON schema as a tool, force tool_choice so it can't
reply in free text, and read the input off the tool_use block.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL, get_client

EXTRACT_SHIPMENT_TOOL = {
    "name": "record_shipment_delay",
    "description": "Record structured details about a shipment delay mentioned in text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "shipment_id": {"type": "string"},
            "delay_days": {"type": "integer"},
            "reason": {"type": "string"},
            "carrier": {"type": ["string", "null"]},
        },
        "required": ["shipment_id", "delay_days", "reason"],
    },
}


def extract_shipment_delay(text: str) -> dict:
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        tools=[EXTRACT_SHIPMENT_TOOL],
        tool_choice={"type": "tool", "name": "record_shipment_delay"},
        messages=[{"role": "user", "content": text}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("Model did not return a tool_use block")


if __name__ == "__main__":
    sample = (
        "Shipment SH-4471 from Meridian Freight is running 3 days late "
        "due to a driver shortage at the origin terminal."
    )
    print(json.dumps(extract_shipment_delay(sample), indent=2))
