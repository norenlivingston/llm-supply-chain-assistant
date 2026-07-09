"""Session 2 - ChromaDB intro using an in-memory, non-persistent collection."""
import chromadb


def run_demo() -> None:
    client = chromadb.Client()  # in-memory only, nothing written to disk
    collection = client.create_collection("scm_terms_demo")

    collection.add(
        ids=["1", "2", "3"],
        documents=[
            "Safety stock is buffer inventory held to protect against demand variability.",
            "The bullwhip effect describes amplified demand variability upstream in a supply chain.",
            "Incoterms define which party bears cost and risk at each stage of a shipment.",
        ],
        metadatas=[
            {"topic": "inventory"},
            {"topic": "demand planning"},
            {"topic": "logistics"},
        ],
    )

    results = collection.query(
        query_texts=["Why does demand variability grow upstream?"], n_results=1
    )
    print(results["documents"][0][0])


if __name__ == "__main__":
    run_demo()
