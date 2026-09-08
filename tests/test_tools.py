"""Tests for session4/tools.py - no API key needed, pure Python + SQLite.

These formalize checks that were previously only run ad hoc: the reorder
point formula, shipment lookups, and specifically the two-layer write
guardrail on flag_shipment_for_expedite (the exact bypass this project's
README documents finding and fixing).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from session4.db import reset_db
from session4.tools import calculate_reorder_point, flag_shipment_for_expedite, lookup_shipment_status


@pytest.fixture(autouse=True)
def clean_db():
    reset_db()
    yield


def test_calculate_reorder_point():
    result = calculate_reorder_point(avg_daily_demand=120, lead_time_days=5, safety_stock=200)
    assert result["reorder_point"] == 800


def test_calculate_reorder_point_no_safety_stock():
    result = calculate_reorder_point(avg_daily_demand=10, lead_time_days=3)
    assert result["reorder_point"] == 30


def test_lookup_shipment_status_found():
    result = lookup_shipment_status("sh-1002")
    assert result["shipment_id"] == "SH-1002"
    assert "carrier" in result


def test_lookup_shipment_status_not_found():
    result = lookup_shipment_status("SH-9999")
    assert "error" in result


def test_expedite_cold_confirm_is_rejected():
    """A first call straight to confirm=True, with no prior preview, must
    not write. This is the exact bypass the guardrail was hardened against."""
    result = flag_shipment_for_expedite("SH-1001", "customer escalating", confirm=True)
    assert "error" in result
    assert lookup_shipment_status("SH-1001")["expedite_requested"] == 0


def test_expedite_preview_does_not_write():
    result = flag_shipment_for_expedite("SH-1002", "customer escalating", confirm=False)
    assert result["status"] == "confirmation_required"
    assert lookup_shipment_status("SH-1002")["expedite_requested"] == 0


def test_expedite_preview_then_confirm_writes():
    flag_shipment_for_expedite("SH-1002", "customer escalating", confirm=False)
    result = flag_shipment_for_expedite("SH-1002", "customer escalating", confirm=True)
    assert result["status"] == "flagged"
    assert lookup_shipment_status("SH-1002")["expedite_requested"] == 1


def test_expedite_unknown_shipment():
    result = flag_shipment_for_expedite("SH-0000", "test", confirm=False)
    assert "error" in result
