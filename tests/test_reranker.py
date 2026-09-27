import json

from src.retrieval.reranker import Reranker


with open("data/test_chunks.json", "r") as file:
    chunks = json.load(file)


documents = []

for chunk in chunks:
    documents.append(
        chunk["description"] + " " + chunk["code"]
    )


query = "calculate average numbers"


results = []

for index, document in enumerate(documents):
    results.append({
        "index": index,
        "document": document
    })


reranker = Reranker()

reranked_results = reranker.rerank(
    query,
    results,
    top_k=5
)


for rank, result in enumerate(reranked_results, start=1):
    print(f"\nRank {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Index: {result['index']}")
    print(f"Document: {result['document']}")