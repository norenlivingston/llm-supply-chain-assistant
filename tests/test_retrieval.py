"""Tests for session3 ingestion + retrieval - no API key needed.

Embedding is local (chromadb's default model), so this only costs the
one-time model download, not an API call. Session-scoped: ingestion only
needs to run once for all tests in this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from session3.ingest import ingest_docs
from session3.retrieval import retrieve


@pytest.fixture(scope="module", autouse=True)
def ingested():
    count = ingest_docs()
    assert count > 0
    yield count


def test_retrieve_returns_results():
    hits = retrieve("What is the bullwhip effect?", k=4)
    assert len(hits) > 0


def test_retrieve_finds_relevant_doc_by_meaning():
    """No word in this query appears verbatim in bullwhip_effect.txt's
    title - this is checking semantic retrieval works, not keyword match."""
    hits = retrieve("Why does demand variability grow upstream in a supply chain?", k=3)
    sources = [hit["source"] for hit in hits]
    assert "bullwhip_effect.txt" in sources


def test_retrieve_similarity_scores_are_ordered():
    hits = retrieve("What is EOQ?", k=4)
    similarities = [hit["similarity"] for hit in hits]
    assert similarities == sorted(similarities, reverse=True)
