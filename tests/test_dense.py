import json

from src.retrieval.dense import DenseRetriever


with open("data/test_chunks.json", "r") as file:
    chunks = json.load(file)


documents = []

for chunk in chunks:
    documents.append(
        chunk["description"] + " " + chunk["code"]
    )


retriever = DenseRetriever()

query = "calculate the mean of several values"

results = retriever.search(
    query,
    documents,
    top_k=5
)


for rank, result in enumerate(results, start=1):
    print(f"\nRank {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {result['document']}")