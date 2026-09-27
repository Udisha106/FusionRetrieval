"""
load_mteb_eval_set.py  (FIXED)
--------------------------------
Verified against your actual installed mteb version (2.21.6) by inspecting
the real object structure directly. The real path is:

    task.dataset["default"]["test"]["queries"]
    task.dataset["default"]["test"]["corpus"]
    task.dataset["default"]["test"]["relevant_docs"]

Each of queries/corpus is likely a HuggingFace `Dataset` object (rows with
an "id" field and a "text" field) rather than a plain Python dict, so this
version converts them to plain dicts defensively either way.

Usage:
    python load_mteb_eval_set.py
"""

from typing import List, Tuple

TASK_NAME = "AppsRetrieval"
SUBSET = "default"   # confirmed from your dir() output
SPLIT = "test"        # confirmed from your dir() output


def load_mteb_task(task_name: str = TASK_NAME):
    import mteb
    task = mteb.get_task(task_name)
    task.load_data()
    return task


def _to_id_text_dict(hf_dataset_or_dict) -> dict:
    """
    Handles both shapes:
      - a HuggingFace Dataset with rows like {"id": ..., "text": ...}
      - a plain dict already shaped like {id: text}
    Returns a plain {id: text} dict either way.
    """
    if isinstance(hf_dataset_or_dict, dict):
        # Already id->text, or id->{"text": ...}
        first_val = next(iter(hf_dataset_or_dict.values())) if hf_dataset_or_dict else None
        if isinstance(first_val, dict) and "text" in first_val:
            return {k: v["text"] for k, v in hf_dataset_or_dict.items()}
        return hf_dataset_or_dict

    # Assume HF Dataset: iterate rows
    result = {}
    for row in hf_dataset_or_dict:
        row_id = row.get("id") or row.get("_id")
        text = row.get("text")
        if row_id is not None:
            result[row_id] = text
    return result


def to_benchmark_eval_set(task, max_queries: int = 50) -> List[Tuple[str, set]]:
    split_data = task.dataset[SUBSET][SPLIT]

    queries = _to_id_text_dict(split_data["queries"])
    relevant_docs = split_data["relevant_docs"]  # {query_id: {doc_id: score}}
    if not isinstance(relevant_docs, dict):
        # in case it's also a Dataset-like structure, normalize it
        relevant_docs = {row["query-id"]: {row["corpus-id"]: row.get("score", 1)}
                          for row in relevant_docs}

    eval_set = []
    for i, (qid, query_text) in enumerate(queries.items()):
        if i >= max_queries:
            break
        rel = relevant_docs.get(qid, {})
        relevant_ids = set(rel.keys()) if isinstance(rel, dict) else set()
        if relevant_ids:
            eval_set.append((query_text, relevant_ids))

    return eval_set


def get_corpus_lookup(task) -> dict:
    split_data = task.dataset[SUBSET][SPLIT]
    return _to_id_text_dict(split_data["corpus"])


if __name__ == "__main__":
    print(f"Loading MTEB task: {TASK_NAME} ...")
    task = load_mteb_task(TASK_NAME)

    eval_set = to_benchmark_eval_set(task, max_queries=10)
    corpus = get_corpus_lookup(task)

    print(f"\nLoaded {len(eval_set)} labeled queries and {len(corpus)} corpus documents.\n")
    print("Sample queries:")
    for query, relevant_ids in eval_set[:5]:
        preview = query[:80] if query else "(empty)"
        print(f"  '{preview}...' -> relevant doc ids: {relevant_ids}")
