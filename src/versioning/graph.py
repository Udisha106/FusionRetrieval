"""
graph.py
---------
Builds an evolution graph across commits: nodes are chunk@commit,
edges represent how a chunk evolved over time (derived_from, moved_from,
modified_from). Uses content_hash + function/file identity to link
chunks across versions without needing embeddings.
"""

import networkx as nx


class EvolutionGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def _node_id(self, chunk, commit_hash):
        return chunk["chunk_id"] + "@" + commit_hash[:8]

    def add_commit_chunks(self, chunks, commit_hash):
        for chunk in chunks:
            node_id = self._node_id(chunk, commit_hash)
            self.graph.add_node(
                node_id,
                chunk_id=chunk["chunk_id"],
                commit_hash=commit_hash,
                file_path=chunk.get("file_path"),
                function_name=chunk.get("function_name"),
                content_hash=chunk.get("content_hash"),
            )

    def link_commits(self, old_chunks, old_commit_hash, new_chunks, new_commit_hash):
        old_by_id = {c["chunk_id"]: c for c in old_chunks}

        for new_chunk in new_chunks:
            cid = new_chunk["chunk_id"]
            new_node = self._node_id(new_chunk, new_commit_hash)

            if cid in old_by_id:
                old_chunk = old_by_id[cid]
                old_node = self._node_id(old_chunk, old_commit_hash)

                if old_chunk["content_hash"] == new_chunk["content_hash"]:
                    relation = "unchanged_from"
                else:
                    relation = "modified_from"

                self.graph.add_edge(new_node, old_node, relation=relation)
            else:
                same_function_matches = [
                    c for c in old_chunks
                    if c.get("function_name") == new_chunk.get("function_name")
                    and c.get("function_name") is not None
                    and c["chunk_id"] != cid
                ]
                if same_function_matches:
                    old_node = self._node_id(same_function_matches[0], old_commit_hash)
                    self.graph.add_edge(new_node, old_node, relation="moved_from")

    def get_lineage(self, chunk_id, commit_hash):
        start_node = chunk_id + "@" + commit_hash[:8]
        lineage = [start_node]
        visited = {start_node}
        current = start_node
        while True:
            successors = list(self.graph.successors(current))
            if not successors:
                break
            current = successors[0]
            if current in visited:
                break
            visited.add(current)
            lineage.append(current)
        return lineage

    def num_nodes(self):
        return self.graph.number_of_nodes()

    def num_edges(self):
        return self.graph.number_of_edges()


if __name__ == "__main__":
    old_chunks = [
        {"chunk_id": "auth_verify_1", "file_path": "auth.py", "function_name": "verify_token", "content_hash": "hash_v1"},
        {"chunk_id": "auth_login_2", "file_path": "auth.py", "function_name": "login", "content_hash": "hash_login_v1"},
    ]

    new_chunks = [
        {"chunk_id": "auth_verify_1", "file_path": "auth.py", "function_name": "verify_token", "content_hash": "hash_v2"},
        {"chunk_id": "auth_login_2", "file_path": "auth.py", "function_name": "login", "content_hash": "hash_login_v1"},
    ]

    eg = EvolutionGraph()
    eg.add_commit_chunks(old_chunks, "commitA1234567")
    eg.add_commit_chunks(new_chunks, "commitB7654321")
    eg.link_commits(old_chunks, "commitA1234567", new_chunks, "commitB7654321")

    print("Nodes:", eg.num_nodes())
    print("Edges:", eg.num_edges())
    print("Lineage of auth_verify_1 at commitB:", eg.get_lineage("auth_verify_1", "commitB7654321"))