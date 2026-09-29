FusionRetrieval



Given a library of code and a query in natural language, FusionRetrieval returns a ranked list of the most relevant code snippets. It uses AST aware chunking, hybrid dense plus BM25 retrieval, query expansion, HyDE, reranking, and MMR diversity filtering. It also tracks how code evolves across git commits, so retrieval results can show a chunk's version history, not just its current form.



ARCHITECTURE



The pipeline works in this order. First, Query Understanding extracts intent, entities, and identifiers from the natural language query. Second, Query Expansion generates related technical terminology. Third, HyDE generates a hypothetical code snippet for the query. Fourth, Multi view Retrieval runs Dense and BM25 search separately for each query view (original, expanded, HyDE). Fifth, RRF Fusion combines all results into a top 50 candidate pool. Sixth, Cross encoder Reranking narrows the top 50 down to a top 20. Seventh, MMR Diversity Filtering selects a final top 10, avoiding redundant results.



Code is indexed using Tree sitter AST parsing, chunked at the function or method level, and hashed with SHA256 for incremental re indexing across commits.



PROJECT STRUCTURE



src/indexing/ contains AST parsing, chunking, metadata, hashing. This is Member 1's work.

src/query/ contains query understanding, expansion, HyDE. This is Member 2's work.

src/retrieval/ contains dense retrieval, BM25, fusion, reranker, MMR. This is Member 3's work.

src/versioning/ contains git history, incremental indexing, evolution graph, lineage. This is Member 4's work.

src/evaluation/ contains metrics, evaluation harness, experiments. This is Member 4's work.

data/chunks/ contains the indexed code chunks file, chunks.jsonl.

demo/app.py is the Streamlit demo UI, built by Member 4.

evaluation/evaluate\_retrieval.py is a quick sanity check evaluation script.



INSTALLATION



Requires Python 3.10 or higher.



Clone the repository and install dependencies using these commands.

git clone://github.com/khyati-ctrl/FusionRetrieval.git

cd FusionRetrieval

pip install sentence\_transformers rank\_bm25 networkx streamlit



DATASET AND INDEXING SETUP



Code chunks are pre indexed and stored at data/chunks/chunks.jsonl. Each chunk contains fields such as chunk\_id, file\_path, language, class\_name, function\_name, parameters, docstring, calls, code, start\_line, end\_line, augmented\_text, and content\_hash, which is a SHA256 hash of the code.



To re index a repository from scratch, run the indexing pipeline in src/indexing/pipeline.py.



RUNNING EVALUATION



Quick sanity check, using 4 hand picked queries, scored with Recall at 1, Recall at 2, and MRR. Run this command.

python -m evaluation.evaluate\_retrieval



Full evaluation harness, with NDCG at 10, MRR, and latency breakdown. Run this command.

python -m src.evaluation.evaluate



Ablation experiments, comparing Dense only, BM25 only, RRF fusion, multi view, plus reranker, plus MMR. Run this command.

python -m src.evaluation.experiments



VERSIONING MODULE USAGE



To read git commit history, run this command.

python -m src.versioning.git\_history



To test incremental indexing using hash based diffing, run this command.

python -m src.versioning.incremental



To build and query the evolution graph, run this command.

python -m src.versioning.graph



For full lineage tracking, which combines git history and the graph, run this command.

python -m src.versioning.lineage



RUNNING THE DEMO



Run this command.

python -m streamlit run demo/app.py



This opens a browser UI where you can type a natural language question about the codebase and see the following. Query understanding, showing intent, entities, and identifiers. Vocabulary bridge, showing expanded query terms. HyDE, showing a hypothetical code snippet. Ranked retrieved code with scores. Evolution and lineage view, currently demoed on simulated version history, though the underlying pipeline works identically on real multi commit repos. Per stage latency breakdown.



TEAM



Member 1 handled code indexing, meaning AST parsing, chunking, metadata, and hashing.

Member 2 handled query intelligence, meaning analysis, expansion, and HyDE.

Member 3 handled retrieval and ranking, meaning dense retrieval, BM25, fusion, reranker, and MMR.

Member 4 handled versioning, evaluation, and the demo.

