"""
incremental.py
----------------
Handles incremental indexing: given a set of chunks (from Member 1's
indexing pipeline), figure out which chunks are NEW, which are UNCHANGED
(reuse existing embedding), and which are CHANGED (need re-embedding).

This is based purely on content_hash comparison, so it doesn't need
embeddings or any other teammate's code to work or be tested.
"""

import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ChunkDiff:
    new_chunks: List[dict]
    changed_chunks: List[dict]
    unchanged_chunks: List[dict]
    removed_chunk_ids: List[str]


class IncrementalIndexer:
    def __init__(self):
        # cache maps chunk_id -> content_hash, representing "what we've
        # already indexed / embedded before"
        self.cache: Dict[str, str] = {}

    def load_cache_from_chunks(self, chunks: List[dict]):
        """Treat a given chunk list as the 'previous' state of the index."""
        self.cache = {c["chunk_id"]: c["content_hash"] for c in chunks}

    def diff(self, new_chunk_list: List[dict]) -> ChunkDiff:
        """
        Compare new_chunk_list against self.cache and classify each chunk.
        """
        new_chunks = []
        changed_chunks = []
        unchanged_chunks = []

        seen_ids = set()

        for chunk in new_chunk_list:
            cid = chunk["chunk_id"]
            chash = chunk["content_hash"]
            seen_ids.add(cid)

            if cid not in self.cache:
                new_chunks.append(chunk)
            elif self.cache[cid] != chash:
                changed_chunks.append(chunk)
            else:
                unchanged_chunks.append(chunk)

        removed_chunk_ids = [
            cid for cid in self.cache.keys() if cid not in seen_ids
        ]

        return ChunkDiff(
            new_chunks=new_chunks,
            changed_chunks=changed_chunks,
            unchanged_chunks=unchanged_chunks,
            removed_chunk_ids=removed_chunk_ids,
        )

    def update_cache(self, new_chunk_list: List[dict]):
        """Call this after processing a diff, to make the new state the baseline."""
        self.cache = {c["chunk_id"]: c["content_hash"] for c in new_chunk_list}

    def stats_summary(self, diff: ChunkDiff) -> dict:
        return {
            "new": len(diff.new_chunks),
            "changed": len(diff.changed_chunks),
            "unchanged": len(diff.unchanged_chunks),
            "removed": len(diff.removed_chunk_ids),
            "total_needing_embedding": len(diff.new_chunks) + len(diff.changed_chunks),
        }


def load_chunks_jsonl(path: str) -> List[dict]:
    """Utility to load a chunks.jsonl file (Member 1's output format)."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    # Real test using Member 1's actual chunks.jsonl
    chunks = load_chunks_jsonl("data/chunks/chunks.jsonl")

    indexer = IncrementalIndexer()

    # Simulate "commit A": treat current chunks as the old/cached state
    indexer.load_cache_from_chunks(chunks)

    # Simulate "commit B": pretend one chunk's code changed
    modified_chunks = [dict(c) for c in chunks]  # copy
    modified_chunks[0]["content_hash"] = "CHANGED_HASH_EXAMPLE"

    # Also simulate a brand new chunk being added
    modified_chunks.append({
        "chunk_id": "test_new_function_99",
        "content_hash": "NEW_HASH_EXAMPLE",
    })

    diff = indexer.diff(modified_chunks)
    print("Diff results:")
    print(f"  New:       {[c['chunk_id'] for c in diff.new_chunks]}")
    print(f"  Changed:   {[c['chunk_id'] for c in diff.changed_chunks]}")
    print(f"  Unchanged: {[c['chunk_id'] for c in diff.unchanged_chunks]}")
    print(f"  Removed:   {diff.removed_chunk_ids}")
    print()
    print("Summary:", indexer.stats_summary(diff))