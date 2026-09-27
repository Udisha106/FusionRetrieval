import json

from src.retrieval.pipeline import RetrievalPipeline


with open("data/test_chunks.json", "r") as file:
    chunks = json.load(file)


documents = []

for chunk in chunks:
    documents.append(
        chunk["description"] + " " + chunk["code"]
    )


pipeline = RetrievalPipeline(documents)

query = "calculate average numbers"

results = pipeline.search(
    query,
    rrf_top_k=5,
    rerank_top_k=3,
    final_top_k=2
)


print("\n===== FINAL RESULTS =====")

for rank, result in enumerate(results, start=1):
    print(
        f"\nRank {rank}"
    )
    print(
        f"Score: {result['score']:.4f}"
    )
    print(
        f"Index: {result['index']}"
    )
    print(
        f"Document: {result['document']}"
    )