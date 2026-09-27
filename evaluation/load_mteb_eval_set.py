"""
load_mteb_eval_set.py
----------------------
Loads the real, already-labeled eval data straight from MTEB -- no manual
labeling needed. This replaces the "build your own eval set by hand" plan
now that we know you're using mteb.

Requires: pip install mteb
(needs internet access to download the dataset the first time you run it)

Usage:
    python3 load_mteb_eval_set.py

What this gives you:
    - task.queries        -> dict of {query_id: query_text}
    - task.corpus         -> dict of {doc_id: {"text": code_snippet, ...}}
    - task.relevant_docs  -> dict of {query_id: {doc_id: relevance_score}}
    This is the SAME thing you would have hand-built in eval_set.csv --
    except it's real, already labeled, and much bigger.
"""

from typing import List, Tuple

TASK_NAME = "AppsRetrieval"  # <-- change this if your task name is different


def load_mteb_task(task_name: str = TASK_NAME):
    import mteb  # imported here so the rest of the file works even if mteb isn't installed yet

    task = mteb.get_task(task_name)
    task.load_data()
    return task


def to_benchmark_eval_set(task, split: str = "test", max_queries: int = 50) -> List[Tuple[str, set]]:
    """
    Converts MTEB's format into the (query_text, {relevant_doc_ids}) format
    that benchmark.py's run() function already expects.

    max_queries caps how many queries you evaluate on -- start small (e.g. 50)
    for fast iteration, raise it later once your pipeline is stable.
    """
    queries = task.queries[split] if isinstance(task.queries, dict) and split in task.queries else task.queries
    relevant = task.relevant_docs[split] if isinstance(task.relevant_docs, dict) and split in task.relevant_docs else task.relevant_docs

    eval_set = []
    for i, (qid, query_text) in enumerate(queries.items()):
        if i >= max_queries:
            break
        relevant_ids = set(relevant.get(qid, {}).keys())
        if relevant_ids:  # skip queries with no labeled answer
            eval_set.append((query_text, relevant_ids))

    return eval_set


def get_corpus_lookup(task, split: str = "test") -> dict:
    """
    Returns {doc_id: code_text} so your search_fn can actually embed/search
    the real corpus, not a toy one.
    """
    corpus = task.corpus[split] if isinstance(task.corpus, dict) and split in task.corpus else task.corpus
    return {doc_id: doc.get("text", doc) if isinstance(doc, dict) else doc
            for doc_id, doc in corpus.items()}


if __name__ == "__main__":
    print(f"Loading MTEB task: {TASK_NAME} ...")
    task = load_mteb_task(TASK_NAME)

    eval_set = to_benchmark_eval_set(task, max_queries=10)
    corpus = get_corpus_lookup(task)

    print(f"\nLoaded {len(eval_set)} labeled queries and {len(corpus)} corpus documents.\n")
    print("Sample queries:")
    for query, relevant_ids in eval_set[:5]:
        print(f"  '{query[:80]}...' -> relevant doc ids: {relevant_ids}")
