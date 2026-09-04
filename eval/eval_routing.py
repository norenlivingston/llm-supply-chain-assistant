"""Agent tool-routing eval.

Not a RAG-groundedness eval (whether the answer text is correct) - this
scores whether session4/agent.py's routing logic calls the *right tool(s)*
for each question. Routing correctness is the specific thing an agentic
layer is supposed to add over plain RAG, so it's the specific thing worth
having checked-in evidence for, not just a demo.

Requires ANTHROPIC_API_KEY - each case makes real API calls. Run docs
ingestion first (python -m session3.ingest) so search_knowledge_base has
something to retrieve.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session4.agent import run_agent
from session4.db import reset_db

CASES = [
    {
        "name": "concept_question",
        "question": "What is the bullwhip effect?",
        "expected_tools": {"search_knowledge_base"},
    },
    {
        "name": "math_question",
        "question": (
            "What's the reorder point if average daily demand is 120 units, "
            "lead time is 5 days, and safety stock is 200 units?"
        ),
        "expected_tools": {"calculate_reorder_point"},
    },
    {
        "name": "shipment_lookup",
        "question": "Where is shipment SH-1002 right now?",
        "expected_tools": {"lookup_shipment_status"},
    },
    {
        "name": "combined_kb_and_shipment",
        "question": (
            "Given what causes the bullwhip effect, what's the status of "
            "SH-1001 and should I be worried about a stockout?"
        ),
        "expected_tools": {"search_knowledge_base", "lookup_shipment_status"},
    },
    {
        "name": "expedite_request_previews_only",
        "question": "Flag shipment SH-1002 for expedite, the customer is escalating.",
        "expected_tools": {"flag_shipment_for_expedite"},
    },
    {
        "name": "no_tool_needed",
        "question": "Hi, what can you help me with?",
        "expected_tools": set(),
    },
]


def score_case(case: dict) -> dict:
    result = run_agent(case["question"], verbose=False)
    called_tools = {step["tool"] for step in result["trace"]}
    return {
        "name": case["name"],
        "question": case["question"],
        "expected": sorted(case["expected_tools"]),
        "called": sorted(called_tools),
        "passed": called_tools == case["expected_tools"],
        "answer": result["answer"],
    }


def run_eval() -> list[dict]:
    reset_db()
    results = [score_case(case) for case in CASES]

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['name']}: {r['question']}")
        print(f"    expected tools: {r['expected']}")
        print(f"    called tools:   {r['called']}")

    passed_count = sum(r["passed"] for r in results)
    print(f"\n{passed_count}/{len(results)} routing cases passed")
    return results


def score_confirm_flow() -> dict:
    """Verifies the expedite guardrail end to end: the first turn must
    preview only (no DB write), and only the confirmed follow-up turn may
    apply it."""
    reset_db()
    from session4.db import get_connection

    turn1 = run_agent(
        "Flag shipment SH-1003 for expedite, the customer is escalating.",
        verbose=False,
    )
    turn1_tools = [step["tool"] for step in turn1["trace"]]
    turn1_wrote = any(
        step["tool"] == "flag_shipment_for_expedite" and step["output"].get("status") == "flagged"
        for step in turn1["trace"]
    )

    conn = get_connection()
    row_after_turn1 = conn.execute(
        "SELECT expedite_requested FROM shipments WHERE shipment_id = 'SH-1003'"
    ).fetchone()
    conn.close()

    turn2 = run_agent("Yes, go ahead.", history=turn1["messages"], verbose=False)
    turn2_wrote = any(
        step["tool"] == "flag_shipment_for_expedite" and step["output"].get("status") == "flagged"
        for step in turn2["trace"]
    )

    conn = get_connection()
    row_after_turn2 = conn.execute(
        "SELECT expedite_requested FROM shipments WHERE shipment_id = 'SH-1003'"
    ).fetchone()
    conn.close()

    passed = (
        not turn1_wrote
        and row_after_turn1["expedite_requested"] == 0
        and turn2_wrote
        and row_after_turn2["expedite_requested"] == 1
    )

    result = {
        "name": "confirm_then_act_guardrail",
        "turn1_tools_called": turn1_tools,
        "turn1_wrote_to_db": turn1_wrote,
        "turn2_wrote_to_db": turn2_wrote,
        "passed": passed,
    }
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] confirm_then_act_guardrail")
    print(f"    turn 1 tools called: {turn1_tools} (must NOT write)")
    print(f"    turn 1 wrote to db:  {turn1_wrote}")
    print(f"    turn 2 wrote to db:  {turn2_wrote} (must write, after user confirmed)")
    return result


if __name__ == "__main__":
    run_eval()
    score_confirm_flow()
