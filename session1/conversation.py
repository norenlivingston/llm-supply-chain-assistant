"""Session 1 - a multi-turn conversation loop that maintains message history."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL, get_client

SYSTEM_PROMPT = "You are a supply chain operations assistant."


def run_conversation() -> None:
    client = get_client()
    messages = []
    print("Supply chain assistant. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        reply = response.content[0].text
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    run_conversation()
