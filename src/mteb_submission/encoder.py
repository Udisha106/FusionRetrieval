"""
encoder.py
-----------
Wraps our team's query pipeline (Member 2: expansion + HyDE) around a
sentence embedding model, in the shape MTEB's AbsEncoder expects.

This is used for the OFFICIAL SCREENING submission: MTEB's encode() only
supports embedding-based ranking (cosine similarity between query and
document vectors). It does NOT support BM25, reranking, or MMR — those
require the raw text of retrieved candidates, which this interface never
exposes. Our full hybrid pipeline (Member 3's BM25 + fusion + reranker +
MMR) is demonstrated separately in the live demo / hands-on evaluation.

Pre-processing applied here (allowed and encouraged by the guidelines):
  - Query expansion (related technical terms)
  - HyDE (hypothetical code snippet)
  These are combined into a single averaged query embedding before
  MTEB computes cosine similarity against the corpus.
"""

import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.query.query_pipeline import QueryPipeline
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType
from mteb.models.model_meta import ModelMeta


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.query_pipeline = QueryPipeline()
        self.mteb_model_meta = ModelMeta(
            name="FusionRetrieval/PrePostPipelineEncoder",
            revision="1",
            loader=None,
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=384,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["Sentence Transformers"],
            reference=None,
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    def _embed_texts(self, texts, batch_size=64):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def _build_query_embedding(self, query_text):
        """
        Runs the query through our pre-processing pipeline (expansion + HyDE),
        embeds all views, and averages them into a single vector.
        """
        mv_query = self.query_pipeline.process(query_text)
        views = mv_query.views_for_retrieval()  # [original, expanded, hyde]

        view_embeddings = self._embed_texts(views)
        combined = np.mean(view_embeddings, axis=0)
        return combined

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        prompt_type: PromptType = None,
        **kwargs,
    ):
        # Flatten the dataloader batches into a plain list of text strings
        texts = []
        for batch in inputs:
            batch_texts = batch["text"] if isinstance(batch, dict) else batch
            texts.extend(batch_texts)

        is_query = prompt_type is not None and prompt_type.value == "query"

        if is_query:
            embeddings = np.array([self._build_query_embedding(t) for t in texts])
        else:
            # Documents/corpus: plain embedding, no query-side preprocessing needed
            embeddings = self._embed_texts(texts, batch_size=kwargs.get("batch_size", 64))

        return embeddings


if __name__ == "__main__":
    # Quick smoke test with a couple of fake sentences, no MTEB needed yet
    encoder = PrePostPipelineEncoder()

    fake_query_emb = encoder._build_query_embedding("how is input validated before saving")
    print("Query embedding shape:", fake_query_emb.shape)

    fake_doc_embs = encoder._embed_texts(["def validate(x): return x is not None", "def save(x): pass"])
    print("Doc embeddings shape:", fake_doc_embs.shape)