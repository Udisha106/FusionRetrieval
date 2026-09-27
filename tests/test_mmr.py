import json

from src.retrieval.mmr import MMR


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


mmr = MMR()

selected_results = mmr.select(
    query,
    results,
    top_k=3
)


print("\n===== MMR RESULTS =====")

for rank, result in enumerate(selected_results, start=1):
    print(
        f"Rank {rank} | "
        f"Index: {result['index']}"
    )
    print(f"Document: {result['document']}")