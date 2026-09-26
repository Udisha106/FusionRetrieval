"""
demo/app.py
------------
Streamlit demo UI for the code retrieval system.
Shows: search box -> query understanding -> HyDE -> ranked results
-> evolution/lineage -> latency breakdown.

All data here is REAL, wired to the actual pipelines built by the team.

Run with: streamlit run demo/app.py
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.query.query_pipeline import QueryPipeline
from src.retrieval.pipeline import RetrievalPipeline


st.set_page_config(page_title="CodeLens", layout="wide")


@st.cache_resource
def load_pipelines():
    with open("data/chunks/chunks.jsonl", "r") as f:
        chunks = [json.loads(line) for line in f]

    documents = [c["augmented_text"] for c in chunks]

    query_pipeline = QueryPipeline()
    retrieval_pipeline = RetrievalPipeline(documents)

    return chunks, query_pipeline, retrieval_pipeline


chunks, query_pipeline, retrieval_pipeline = load_pipelines()

st.title("CodeLens")
st.caption("Ask about the codebase in plain English")

query = st.text_input("Ask about the codebase:", placeholder="e.g. How is the input normalized before saving?")

if st.button("Search") and query.strip():

    timings = {}

    t0 = time.perf_counter()
    mv_query = query_pipeline.process(query)
    timings["Query processing"] = (time.perf_counter() - t0) * 1000

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Query Understanding")
        st.markdown(f"**Intent:** {mv_query.intent}")
        st.markdown(f"**Entities:** {', '.join(mv_query.entities) if mv_query.entities else '-'}")
        st.markdown(f"**Identifiers:** {', '.join(mv_query.identifiers) if mv_query.identifiers else '-'}")

    with col2:
        st.subheader("Vocabulary Bridge")
        st.markdown(f"**Expanded:** {mv_query.expanded}")
        st.subheader("HyDE (Hypothetical Code)")
        st.code(mv_query.hyde, language="python")

    t1 = time.perf_counter()
    results = retrieval_pipeline.search(
        mv_query.views_for_retrieval(),
        rrf_top_k=50,
        rerank_top_k=20,
        final_top_k=10,
    )
    timings["Retrieval (BM25+Dense+Fusion+Rerank+MMR)"] = (time.perf_counter() - t1) * 1000

    st.subheader("Retrieved Code")

    for rank, result in enumerate(results, start=1):
        idx = result["index"]
        chunk = chunks[idx]

        with st.expander(f"#{rank}  {chunk['file_path']}  →  {chunk.get('function_name', '')}", expanded=(rank <= 3)):
            st.code(chunk["code"], language=chunk.get("language", "python"))
            st.markdown(f"**Score:** {result.get('score', 'N/A')}")

    st.subheader("Performance")
    total_latency = sum(timings.values())

    for stage, ms in timings.items():
        st.markdown(f"- **{stage}:** {ms:.2f} ms")
    st.markdown(f"**Total: {total_latency:.2f} ms**")

    st.caption(f"Chunks indexed: {len(chunks)}")

else:
    st.info("Enter a query above and click Search to see results.")