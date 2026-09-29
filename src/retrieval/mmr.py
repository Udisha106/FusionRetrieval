import numpy as np


class MMR:

    def select(
        self,
        query,
        results,
        document_embeddings,
        query_embedding,
        top_k=10,
        lambda_param=0.7
    ):

        if not results:
            return []

        indices = [
            result["index"]
            for result in results
        ]

        candidate_embeddings = document_embeddings[indices]

        relevance_scores = np.dot(
            candidate_embeddings,
            query_embedding
        )

        selected = []
        remaining = list(range(len(results)))

        while remaining and len(selected) < top_k:

            if not selected:

                best_index = max(
                    remaining,
                    key=lambda i: relevance_scores[i]
                )

            else:

                best_index = max(
                    remaining,
                    key=lambda i:
                    lambda_param * relevance_scores[i]
                    - (1 - lambda_param) *
                    max(
                        np.dot(
                            candidate_embeddings[i],
                            candidate_embeddings[j]
                        )
                        for j in selected
                    )
                )

            selected.append(best_index)
            remaining.remove(best_index)

        return [
            results[i]
            for i in selected
        ]