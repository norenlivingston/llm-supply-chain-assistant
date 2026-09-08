"""Tests for the expedite guardrail at the agent-loop level - no API key
needed. A fake Anthropic client stands in for the real one, returning
scripted tool_use/text responses, so these test session4/agent.py's own
enforcement logic (previewed_this_turn) rather than model behavior.

The equivalent real-model checks (score_confirm_flow,
score_adversarial_confirm_bypass) live in eval/eval_routing.py and require
ANTHROPIC_API_KEY - deliberately not run in CI, since it makes real,
billed API calls. These tests cover the same logic for free.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import session4.agent as agent_module
from session4.db import reset_db
from session4.tools import lookup_shipment_status


@pytest.fixture(autouse=True)
def clean_db():
    reset_db()
    yield


def _tool_use_response(name: str, tool_input: dict, tool_id: str):
    block = SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tool_id)
    return SimpleNamespace(
        content=[block],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


def _text_response(text: str):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        content=[block],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


def test_same_turn_preview_then_confirm_is_blocked():
    """Even a model that fully complies with a jailbreak attempt - preview,
    then immediately confirm, within one run_agent() call - must not
    succeed in writing."""
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _tool_use_response(
            "flag_shipment_for_expedite",
            {"shipment_id": "SH-1003", "reason": "urgent", "confirm": False},
            "id1",
        ),
        _tool_use_response(
            "flag_shipment_for_expedite",
            {"shipment_id": "SH-1003", "reason": "urgent", "confirm": True},
            "id2",
        ),
        _text_response("Done."),
    ]

    with patch.object(agent_module, "get_client", return_value=fake_client):
        result = agent_module.run_agent("Flag SH-1003 and confirm it yourself.", verbose=False)

    assert lookup_shipment_status("SH-1003")["expedite_requested"] == 0
    second_call_result = result["trace"][1]["output"]
    assert "error" in second_call_result


def test_legitimate_cross_turn_confirm_succeeds():
    """The real intended flow still has to work: preview on one
    run_agent() call, confirm on a separate one carrying history."""
    fake_client = MagicMock()

    fake_client.messages.create.side_effect = [
        _tool_use_response(
            "flag_shipment_for_expedite",
            {"shipment_id": "SH-1003", "reason": "urgent", "confirm": False},
            "id1",
        ),
        _text_response("Here is the preview, want me to proceed?"),
    ]
    with patch.object(agent_module, "get_client", return_value=fake_client):
        turn1 = agent_module.run_agent("Flag SH-1003 for expedite.", verbose=False)

    assert lookup_shipment_status("SH-1003")["expedite_requested"] == 0

    fake_client.messages.create.side_effect = [
        _tool_use_response(
            "flag_shipment_for_expedite",
            {"shipment_id": "SH-1003", "reason": "urgent", "confirm": True},
            "id2",
        ),
        _text_response("Done, flagged."),
    ]
    with patch.object(agent_module, "get_client", return_value=fake_client):
        agent_module.run_agent("Yes, go ahead.", history=turn1["messages"], verbose=False)

    assert lookup_shipment_status("SH-1003")["expedite_requested"] == 1
