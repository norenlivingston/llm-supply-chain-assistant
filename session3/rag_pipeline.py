"""Session 3 - full RAG pipeline: retrieve, build a grounded prompt, generate with citations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL, get_client
from session3.retrieval import retrieve

SYSTEM_PROMPT = (
    "You are a supply chain knowledge assistant. Answer ONLY using the "
    "provided context excerpts. Cite sources inline like [source_file.txt]. "
    "If the context does not contain enough information to answer, say so "
    "explicitly instead of guessing."
)


def build_prompt(question: str, hits: list[dict]) -> str:
    context_block = "\n\n".join(f"[{hit['source']}]\n{hit['text']}" for hit in hits)
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline citations."
    )


def answer_question(question: str, k: int = 4) -> dict:
    hits = retrieve(question, k=k)
    if not hits:
        return {"answer": "No relevant context found in the knowledge base.", "sources": []}

    prompt = build_prompt(question, hits)
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": response.content[0].text,
        "sources": sorted({hit["source"] for hit in hits}),
    }


if __name__ == "__main__":
    result = answer_question("What is the bullwhip effect and how can it be mitigated?")
    print(result["answer"])
    print("\nSources:", ", ".join(result["sources"]))
