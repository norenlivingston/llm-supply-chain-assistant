# llm-supply-chain-assistant

A RAG pipeline with an agentic tool-calling layer on top, built against the
Anthropic API and ChromaDB, in the supply chain domain. No LangChain — raw
Anthropic tool calling throughout, for a transparent agent loop.

## Structure

```
llm-supply-chain-assistant/
├── docs/            supply chain knowledge base (.txt), ingested into Chroma
├── config.py         shared MODEL constant + client/collection factories
├── session1/          Anthropic API basics: single call, multi-turn loop
├── session2/          temperature control, prompt templates, structured
│                      JSON extraction (forced tool use), in-memory Chroma
├── session3/          persistent Chroma, ingestion, retrieval, full RAG
│                      pipeline with citations + hallucination guardrails,
│                      Streamlit UI
└── session4/          agentic layer: tool definitions, multi-step
                       reasoning/routing loop, Streamlit UI with a visible
                       tool-call trace
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
- **`lookup_shipment_status`** — a structured lookup against a mock
  shipment table, standing in for a real TMS/ERP call. Demonstrates routing
  between unstructured knowledge search and structured system-of-record
  lookups, a distinction real supply chain agents have to make constantly.

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

## Notes

- `chroma_store/` is gitignored and created locally by `session3.ingest`.
- `MODEL` lives in one place (`config.py`) — change it there if your account
  uses a different model id than the one checked in.
