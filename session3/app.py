"""Session 3 - Streamlit front end for the RAG pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from session3.rag_pipeline import answer_question

st.set_page_config(page_title="Supply Chain RAG Assistant", page_icon="📦")
st.title("📦 Supply Chain Knowledge Assistant")
st.caption("Answers are grounded in the ingested knowledge base with source citations.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a supply chain question...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            result = answer_question(question)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
