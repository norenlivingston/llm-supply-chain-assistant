"""Shared constants and client factories used across all sessions."""
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

MODEL = "claude-sonnet-4-6"

DOCS_DIR = ROOT_DIR / "docs"
CHROMA_DIR = ROOT_DIR / "chroma_store"
COLLECTION_NAME = "supply_chain_kb"

# Approximate per-token pricing, used only to show a rough cost estimate in
# the UI. Update these two numbers if your account's actual model pricing
# differs - this is a display estimate, not billing data pulled from
# Anthropic, so it will drift from the truth if left unmaintained.
INPUT_PRICE_PER_MTOK = 3.00  # USD per 1,000,000 input tokens
OUTPUT_PRICE_PER_MTOK = 15.00  # USD per 1,000,000 output tokens


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * INPUT_PRICE_PER_MTOK + (
        output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_MTOK


def get_client():
    import anthropic

    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
