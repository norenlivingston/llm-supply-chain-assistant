# llm-supply-chain-assistant

A RAG pipeline with an agentic tool-calling layer on top, built against the
Anthropic API and ChromaDB, in the supply chain domain. No LangChain — raw
Anthropic tool calling throughout, for a transparent agent loop.

## Structure

```
llm-supply-chain-assistant/
├── docs/            supply chain knowledge base (.txt), ingested into Chroma
├── config.py         shared MODEL constant + client/collection factories
├── app.py             unified demo app: toggle between Plain RAG and
│                      Agentic mode in one place (see below)
├── session1/          Anthropic API basics: single call, multi-turn loop
├── session2/          temperature control, prompt templates, structured
│                      JSON extraction (forced tool use), in-memory Chroma
├── session3/          persistent Chroma, ingestion, retrieval, full RAG
│                      pipeline with citations + hallucination guardrails,
│                      standalone Streamlit UI
├── session4/          agentic layer: tool definitions (including one that
│                      writes, gated behind human confirmation), SQLite
│                      shipment records, multi-step reasoning/routing loop,
│                      standalone Streamlit UI with a visible tool-call trace
└── eval/              tool-routing accuracy eval for the agent (see below)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

## Running each session

```bash
# Session 1
python -m session1.chat
python -m session1.conversation

# Session 2
python -m session2.temperature_demo
python -m session2.prompt_templates
python -m session2.json_extraction
python -m session2.chroma_intro

# Session 3 - ingest once, then query
python -m session3.ingest
python -m session3.retrieval
python -m session3.rag_pipeline
streamlit run session3/app.py

# Session 4 - agentic layer (requires session3.ingest to have run at least once)
python -m session4.agent
streamlit run session4/app.py

# Unified demo app - toggle between Plain RAG and Agentic mode side by side
streamlit run app.py

# Eval - does the agent route to the right tool(s) per question?
python -m eval.eval_routing
```

## Session 4: the agentic layer

`session3` answers questions by always doing the same thing: retrieve, then
generate. `session4` replaces that fixed pipeline with a model-driven loop
that decides, per question, what it actually needs:

- **`search_knowledge_base`** — unstructured retrieval over the RAG
  knowledge base (concepts, definitions, best practices). Wraps
  `session3.retrieval.retrieve` directly; the agent's own reasoning turn
  synthesizes the answer and cites sources, rather than nesting a second
  full RAG completion call.
- **`calculate_reorder_point`** — deterministic inventory math. Forces the
  model to call a tool for arithmetic instead of hallucinating a number.
- **`lookup_shipment_status`** — a structured, read-only lookup against a
  SQLite `shipments` table (`session4/db.py`), standing in for a real
  TMS/ERP call. Demonstrates routing between unstructured knowledge search
  and structured system-of-record reads, a distinction real supply chain
  agents have to make constantly.
- **`flag_shipment_for_expedite`** — the one tool that writes. It defaults
  to `confirm=false`, which looks the shipment up and returns a preview of
  the change without writing anything; the system prompt requires the
  agent to relay that preview and get the user's explicit agreement on a
  later turn before calling it again with `confirm=true` to actually apply
  it. The gate is enforced in the function itself, not just the prompt — a
  first call can never write, regardless of what the model decides to do.
  `run_agent` accepts an optional `history` list so a caller can carry the
  conversation across turns, which is what makes the second, confirmed
  call possible.

`session4/agent.py` implements the loop: call the model with the tool
definitions, and while `stop_reason == "tool_use"`, execute the requested
tool(s), append `tool_result` blocks, and call again — up to a step cap.
The loop ends the moment the model returns a plain text response, so it can
resolve in one step (no tool needed), two (one lookup), or more (chained
tool calls, e.g. "given what causes the bullwhip effect, what's the status
of SH-1001 and should I be worried about a stockout?" — knowledge search
*and* shipment lookup before answering).

`session4/app.py` is a Streamlit front end that shows the full tool-call
trace (tool name, input, and raw output) for each answer, so the agent's
reasoning path is inspectable rather than a black box.

## The unified app

`app.py` (repo root) puts Session 3's plain RAG pipeline and Session 4's
agent behind a single sidebar toggle instead of two separate `streamlit
run` commands. The point is comparison: ask the same question in both
modes and see the difference directly — e.g. "where is shipment SH-1002?"
gets a real structured answer in Agentic mode (`lookup_shipment_status`)
but only a "not in the knowledge base" response in Plain RAG mode, since
plain retrieval has no access to shipment records at all. This is the
front end worth linking as the live demo; `session3/app.py` and
`session4/app.py` stay in place as the incremental, session-by-session
build artifacts.

## Eval: tool-routing accuracy

`eval/eval_routing.py` is not a RAG-groundedness eval (whether answer text
is correct) — it scores whether the agent calls the *right tool(s)* for a
question, since correct routing is specifically what the agentic layer is
supposed to add over plain RAG. Six cases cover: a concept question
(expects `search_knowledge_base`), a math question (expects
`calculate_reorder_point`), a shipment lookup (expects
`lookup_shipment_status`), a combined question needing both knowledge
search and a shipment lookup, an expedite request (expects the guarded
tool to preview only, not write), and a question needing no tool at all.

A second check, `score_confirm_flow`, verifies the write guardrail
end-to-end across two turns: the first turn must leave the database
unmodified, and only the confirmed follow-up turn may apply the change —
checked against the actual SQLite row, not just the tool's return value.

Both require `ANTHROPIC_API_KEY` and make real API calls.

## Notes

- `chroma_store/` and `shipments.db` are gitignored and created locally by
  `session3.ingest` / `session4.db` on first run.
- `MODEL` lives in one place (`config.py`) — change it there if your account
  uses a different model id than the one checked in.
