"""Session 2 - temperature control: same prompt, different sampling temperatures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL, get_client

PROMPT = "Suggest a name for a new warehouse slotting optimization initiative."


def run_at_temperatures(prompt: str, temperatures=(0.0, 0.5, 1.0)) -> None:
    client = get_client()
    for temp in temperatures:
        response = client.messages.create(
            model=MODEL,
            max_tokens=64,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"temperature={temp}: {response.content[0].text.strip()}")


if __name__ == "__main__":
    run_at_temperatures(PROMPT)
