"""Session 4 - agentic loop with raw Anthropic tool calling.

No framework: this is the loop LangChain/agent frameworks wrap. Each turn,
Claude either returns tool_use blocks (we execute them and feed results
back) or a final text block (stop_reason != "tool_use", loop ends). A
step cap prevents runaway loops if the model keeps calling tools.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL, get_client
from session4.tools import TOOLS, TOOL_IMPLEMENTATIONS

SYSTEM_PROMPT = (
    "You are a supply chain operations agent. You have tools for searching "
    "a knowledge base, doing inventory math, and looking up shipment "
    "records. Decide which tool(s) a question needs, call them, and only "
    "answer once you have what you need. If a question needs no tool, "
    "answer directly. If the knowledge base doesn't contain the answer, "
    "say so instead of guessing. Cite knowledge base sources inline as "
    "[source_file.txt] when you use search_knowledge_base results."
)

MAX_STEPS = 6


def run_tool(name: str, tool_input: dict) -> dict:
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return impl(**tool_input)
    except Exception as exc:
        return {"error": str(exc)}


def run_agent(question: str, verbose: bool = True) -> dict:
    client = get_client()
    messages = [{"role": "user", "content": question}]
    trace = []

    for step in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"answer": final_text, "trace": trace, "steps": step + 1}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"[step {step + 1}] calling {block.name}({block.input})")
            result = run_tool(block.name, block.input)
            trace.append({"tool": block.name, "input": block.input, "output": result})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "Stopped after reaching the max reasoning steps without a final answer.",
        "trace": trace,
        "steps": MAX_STEPS,
    }


if __name__ == "__main__":
    demo_questions = [
        "What is the bullwhip effect?",
        "What's the reorder point if average daily demand is 120 units, "
        "lead time is 5 days, and we hold 200 units of safety stock?",
        "Where is shipment SH-1002 right now?",
        "Given what causes the bullwhip effect, what's the status of "
        "SH-1001 and should I be worried about a stockout?",
    ]
    for q in demo_questions:
        print(f"\n=== {q}")
        result = run_agent(q, verbose=True)
        print(result["answer"])
