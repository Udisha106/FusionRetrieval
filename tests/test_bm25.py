import json

from src.retrieval.bm25 import BM25Retriever


with open("data/test_chunks.json", "r") as file:
    chunks = json.load(file)


documents = []

for chunk in chunks:
    documents.append(
        chunk["description"] + " " + chunk["code"]
    )


retriever = BM25Retriever(documents)

query = "average"

results = retriever.search(
    query,
    top_k=5
)


for rank, result in enumerate(results, start=1):
    print(f"\nRank {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {result['document']}")