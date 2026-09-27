"""
run_real_benchmark.py
---------------------
Runs the retrieval benchmark on the full MTEB AppsRetrieval corpus.

Pipeline:
1. Load MTEB AppsRetrieval data
2. Use the full corpus
3. Embed the corpus once
4. Run Original / Expanded / HyDE / Combined query strategies
5. Compare equal-weight and HyDE-weighted fusion

Run from repo root:
    python scripts/run_real_benchmark.py
"""

import sys
import random
from pathlib import Path

# Make sure imports work from the repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.query.query_pipeline import QueryPipeline
from src.query.benchmark import RetrievalBenchmark
from src.query.load_mteb_eval_set_fixed import (
    load_mteb_task,
    to_benchmark_eval_set,
    get_corpus_lookup,
)


# None = use all available MTEB queries
MAX_QUERIES = 3765

# None = use the complete MTEB corpus
CORPUS_SAMPLE_SIZE = None

MODEL_NAME = "all-MiniLM-L6-v2"


def build_corpus_sample(
    corpus: dict,
    eval_set,
    sample_size=None
) -> dict:
    """
    Use the full corpus when sample_size is None.
    Otherwise guarantee correct answers are included
    and fill the remaining documents randomly.
    """

    if sample_size is None:
        return corpus

    required_ids = set()

    for _, relevant_ids in eval_set:
        required_ids.update(relevant_ids)

    required_ids = {
        rid for rid in required_ids
        if rid in corpus
    }

    remaining_ids = [
        cid for cid in corpus.keys()
        if cid not in required_ids
    ]

    extra_needed = max(
        0,
        sample_size - len(required_ids)
    )

    random.seed(42)

    sampled_extra = random.sample(
        remaining_ids,
        min(extra_needed, len(remaining_ids))
    )

    final_ids = list(required_ids) + sampled_extra

    return {
        cid: corpus[cid]
        for cid in final_ids
    }


def main():

    # ---------------------------------------------------------
    # 1. Load embedding model
    # ---------------------------------------------------------

    print(f"Loading embedding model: {MODEL_NAME} ...")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------------
    # 2. Load MTEB task
    # ---------------------------------------------------------

    print("Loading MTEB task data...")

    task = load_mteb_task()

    eval_set = to_benchmark_eval_set(
        task,
        max_queries=MAX_QUERIES
    )

    full_corpus = get_corpus_lookup(task)

    # ---------------------------------------------------------
    # 3. Select corpus
    # ---------------------------------------------------------

    if CORPUS_SAMPLE_SIZE is None:

        print(
            f"Using full MTEB corpus: "
            f"{len(full_corpus)} documents"
        )

    else:

        print(
            f"Sampling corpus down to "
            f"~{CORPUS_SAMPLE_SIZE} docs "
            "(guaranteeing correct answers are included)..."
        )

    corpus_sample = build_corpus_sample(
        full_corpus,
        eval_set,
        CORPUS_SAMPLE_SIZE
    )

    corpus_ids = list(corpus_sample.keys())

    corpus_texts = [
        corpus_sample[cid]
        for cid in corpus_ids
    ]

    print(
        f"Corpus ready: "
        f"{len(corpus_texts)} documents"
    )

    print(
        f"Queries ready: "
        f"{len(eval_set)} queries"
    )

    # ---------------------------------------------------------
    # 4. Embed corpus ONCE
    # ---------------------------------------------------------

    print(
        f"Embedding {len(corpus_texts)} corpus documents "
        "(this may take a while on CPU)..."
    )

    corpus_embeddings = model.encode(
        corpus_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # ---------------------------------------------------------
    # 5. Normalize corpus embeddings
    # ---------------------------------------------------------

    import numpy as np
    from numpy.linalg import norm

    corpus_norms = norm(
        corpus_embeddings,
        axis=1,
        keepdims=True
    )

    corpus_normalized = (
        corpus_embeddings
        / np.clip(corpus_norms, 1e-10, None)
    )

    # ---------------------------------------------------------
    # 6. Query embedding function
    # ---------------------------------------------------------

    def embed_fn(text: str):

        vec = model.encode(
            [text],
            convert_to_numpy=True
        )[0]

        return vec / max(
            norm(vec),
            1e-10
        )

    # ---------------------------------------------------------
    # 7. Search function
    # ---------------------------------------------------------

    def search_fn(query_vector, k: int):

        sims = corpus_normalized @ query_vector

        top_k_idx = np.argsort(-sims)[:k]

        return [
            corpus_ids[i]
            for i in top_k_idx
        ]

    # ---------------------------------------------------------
    # 8. Create query pipeline ONCE
    # ---------------------------------------------------------

    pipeline = QueryPipeline()

    # ---------------------------------------------------------
    # 9. Equal-weight benchmark
    # ---------------------------------------------------------

    print()
    print(
        f"Running benchmark on "
        f"{len(corpus_sample)} corpus documents "
        f"and {len(eval_set)} queries..."
    )

    print()
    print(
        "Original vs Expanded vs HyDE vs "
        "Combined (equal weights)..."
    )

    bench_equal = RetrievalBenchmark(
        embed_fn=embed_fn,
        search_fn=search_fn,
        pipeline=pipeline
    )

    results_equal = bench_equal.run(
        eval_set,
        k=10
    )

    bench_equal.print_report(
        results_equal
    )

    # ---------------------------------------------------------
    # 10. HyDE weighted benchmark
    # ---------------------------------------------------------

    print()
    print(
        "Running again with HyDE weighted 2x "
        "in the fusion..."
    )

    bench_weighted = RetrievalBenchmark(
        embed_fn=embed_fn,
        search_fn=search_fn,
        pipeline=pipeline,
        fusion_weights=[
            1.0,
            1.0,
            2.0
        ]
    )

    results_weighted = bench_weighted.run(
        eval_set,
        k=10
    )

    bench_weighted.print_report(
        results_weighted
    )

    # ---------------------------------------------------------
    # 11. Summary
    # ---------------------------------------------------------

    print()
    print(
        "--- Summary: HyDE weighting comparison ---"
    )

    print(
        f"{'Strategy':<12} "
        f"{'Equal NDCG':>12} "
        f"{'Weighted NDCG':>15}"
    )

    for strategy in [
        "original",
        "expanded",
        "hyde",
        "combined"
    ]:

        print(
            f"{strategy:<12} "
            f"{results_equal[strategy]['ndcg@10']:>12.3f} "
            f"{results_weighted[strategy]['ndcg@10']:>15.3f}"
        )


if __name__ == "__main__":
    main()