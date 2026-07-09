"""Session 3 - retrieval over the persistent Chroma collection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_collection


def retrieve(query: str, k: int = 4) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source"),
                # collection is configured with hnsw:space="cosine", so
                # chroma's "distance" is 1 - cosine_similarity.
                "similarity": 1 - dist,
            }
        )
    return hits


if __name__ == "__main__":
    for hit in retrieve("What causes demand variability to amplify upstream?"):
        print(f"[{hit['similarity']:.3f}] {hit['source']}: {hit['text'][:100]}...")
