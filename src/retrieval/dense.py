from sentence_transformers import SentenceTransformer
import numpy as np


class DenseRetriever:

    def __init__(self, model=None):
        self.model = model or SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.documents = None
        self.document_embeddings = None

    def index(self, documents, batch_size=64):
        self.documents = documents

        self.document_embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def search(self, query, documents=None, top_k=5):

        if documents is not None and self.document_embeddings is None:
            self.index(documents)

        if self.document_embeddings is None:
            raise ValueError("Call index(documents) before search().")

        query_embedding = self.encode([query])[0]

        scores = np.dot(
            self.document_embeddings,
            query_embedding
        )

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append({
                "index": int(index),
                "score": float(scores[index]),
                "document": self.documents[index]
            })

        return results