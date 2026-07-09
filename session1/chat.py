"""Session 1 - a single Anthropic API call with a system prompt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL, get_client

SYSTEM_PROMPT = (
    "You are a supply chain operations assistant. Answer concisely and "
    "flag any assumptions you make about the reader's business context."
)


def ask(question: str) -> str:
    client = get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


if __name__ == "__main__":
    print(ask("What's the difference between EOQ and safety stock?"))
