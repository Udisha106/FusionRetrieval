from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k=20):

        # Only rerank the candidates we actually need.
        candidates = results[:top_k]

        pairs = [
            [query, result["document"]]
            for result in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False
        )

        reranked_results = []

        for result, score in zip(candidates, scores):
            reranked_results.append({
                "index": result["index"],
                "score": float(score),
                "document": result["document"]
            })

        reranked_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked_results