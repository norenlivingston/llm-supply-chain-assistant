"""Shared constants and client factories used across all sessions."""
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

MODEL = "claude-sonnet-4-6"

DOCS_DIR = ROOT_DIR / "docs"
CHROMA_DIR = ROOT_DIR / "chroma_store"
COLLECTION_NAME = "supply_chain_kb"


def get_client():
    import anthropic

    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
