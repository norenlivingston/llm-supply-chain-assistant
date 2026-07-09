"""Session 3 - chunk /docs and ingest into a persistent Chroma collection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DOCS_DIR, get_collection

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def ingest_docs() -> int:
    collection = get_collection()

    ids, documents, metadatas = [], [], []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.stem}-{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk": i})

    if not ids:
        raise RuntimeError(f"No .txt files found in {DOCS_DIR}")

    # Rerunning ingest shouldn't duplicate chunks, so clear existing entries first.
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


if __name__ == "__main__":
    count = ingest_docs()
    print(f"Ingested {count} chunks from {DOCS_DIR}")
