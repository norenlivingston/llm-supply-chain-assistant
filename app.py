"""Unified demo app: toggle between plain RAG (Session 3) and the agentic
tool-calling layer (Session 4) in one place, so the same question can be
asked both ways and compared directly.

session3/app.py and session4/app.py still exist standalone as the
session-by-session build artifacts; this is the "real" front end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from session3.rag_pipeline import answer_question
from session4.agent import run_agent

st.set_page_config(page_title="Supply Chain Assistant", page_icon="📦")
st.title("📦 Supply Chain Assistant")
st.caption(
    "Agentic mode routes each question to knowledge search, inventory "
    "math, or shipment lookup as needed. Plain RAG always retrieves then "
    "answers, no matter the question. Ask the same thing in both modes to "
    "see the difference."
)

with st.sidebar:
    mode = st.radio(
        "Mode",
        ["Agentic (Session 4)", "Plain RAG (Session 3)"],
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.agent_messages = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []


def render_trace(trace: list[dict]) -> None:
    with st.expander(f"Tool calls ({len(trace)})"):
        for step in trace:
            st.markdown(f"**{step['tool']}**({step['input']})")
            st.json(step["output"])


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            st.caption(turn["mode"])
        st.markdown(turn["content"])
        if turn.get("trace"):
            render_trace(turn["trace"])
        if turn.get("sources"):
            st.caption("Sources: " + ", ".join(turn["sources"]))

question = st.chat_input("Ask about supply chain concepts, math, or a shipment ID...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        st.caption(mode)
        with st.spinner("Thinking..."):
            if mode.startswith("Agentic"):
                result = run_agent(
                    question, history=st.session_state.agent_messages, verbose=False
                )
                st.session_state.agent_messages = result["messages"]
                answer, trace, sources = result["answer"], result["trace"], None
            else:
                result = answer_question(question)
                answer, trace, sources = result["answer"], None, result["sources"]

        st.markdown(answer)
        if trace:
            render_trace(trace)
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
            "mode": mode,
            "trace": trace,
            "sources": sources,
        }
    )
