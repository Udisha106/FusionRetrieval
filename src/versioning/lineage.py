"""
lineage.py
-----------
High-level entry point that ties git_history.py and graph.py together.
Given a repo, walks its commit history, builds the evolution graph
automatically, and exposes a simple function to get the full lineage
of any chunk (current -> previous -> original).

This is the function the API/demo layer should call.
"""

from .git_history import GitHistory
from .graph import EvolutionGraph
from .incremental import load_chunks_jsonl


class LineageTracker:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        self.git_history = GitHistory(repo_path)
        self.graph = EvolutionGraph()

    def build_from_chunk_snapshots(self, snapshots):
        """
        snapshots: list of (commit_hash, chunk_list) tuples, oldest first.
        Each chunk_list is what load_chunks_jsonl() returns for that commit.
        """
        if not snapshots:
            return

        first_commit, first_chunks = snapshots[0]
        self.graph.add_commit_chunks(first_chunks, first_commit)

        for i in range(1, len(snapshots)):
            prev_commit, prev_chunks = snapshots[i - 1]
            curr_commit, curr_chunks = snapshots[i]

            self.graph.add_commit_chunks(curr_chunks, curr_commit)
            self.graph.link_commits(prev_chunks, prev_commit, curr_chunks, curr_commit)

    def get_full_lineage(self, chunk_id, commit_hash):
        """
        Returns the lineage list from current -> original for a chunk,
        with readable labels (current, previous, ... original).
        """
        raw_lineage = self.graph.get_lineage(chunk_id, commit_hash)

        labeled = []
        for idx, node_id in enumerate(raw_lineage):
            if idx == 0:
                label = "current"
            elif idx == len(raw_lineage) - 1:
                label = "original"
            else:
                label = "previous_" + str(idx)
            labeled.append({"node": node_id, "label": label})

        return labeled

    def summary(self):
        return {
            "total_nodes": self.graph.num_nodes(),
            "total_edges": self.graph.num_edges(),
        }


if __name__ == "__main__":
    # Simulate two "snapshots" using the SAME real chunks.jsonl file,
    # pretending one chunk changed between them, just like we did in
    # incremental.py's test. This proves the whole chain works together:
    # git_history + real chunk data + graph + lineage.

    chunks_v1 = load_chunks_jsonl("data/chunks/chunks.jsonl")

    chunks_v2 = [dict(c) for c in chunks_v1]
    chunks_v2[0]["content_hash"] = "SIMULATED_NEW_HASH_FOR_V2"

    tracker = LineageTracker(repo_path=".")

    snapshots = [
        ("commitAAAAAAAA", chunks_v1),
        ("commitBBBBBBBB", chunks_v2),
    ]

    tracker.build_from_chunk_snapshots(snapshots)

    print("Graph summary:", tracker.summary())

    target_chunk_id = chunks_v1[0]["chunk_id"]
    lineage = tracker.get_full_lineage(target_chunk_id, "commitBBBBBBBB")

    print("Lineage of", target_chunk_id, ":")
    for entry in lineage:
        print(" ", entry["label"], "->", entry["node"])