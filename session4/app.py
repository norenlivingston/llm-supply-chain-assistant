"""Session 4 - Streamlit front end for the agent, with a visible tool-call trace."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from session4.agent import run_agent

st.set_page_config(page_title="Supply Chain Agent", page_icon="🤖")
st.title("🤖 Supply Chain Agent")
st.caption(
    "Routes each question to knowledge search, calculation, or shipment "
    "lookup tools as needed."
)

if "history" not in st.session_state:
    st.session_state.history = []


def render_trace(trace: list[dict]) -> None:
    with st.expander(f"Tool calls ({len(trace)})"):
        for step in trace:
            st.markdown(f"**{step['tool']}**({step['input']})")
            st.json(step["output"])


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("trace"):
            render_trace(turn["trace"])

question = st.chat_input("Ask about supply chain concepts, math, or a shipment ID...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            result = run_agent(question, verbose=False)
        st.markdown(result["answer"])
        if result["trace"]:
            render_trace(result["trace"])

    st.session_state.history.append(
        {"role": "assistant", "content": result["answer"], "trace": result["trace"]}
    )
