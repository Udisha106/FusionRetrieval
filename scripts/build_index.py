"""
build_index.py
----------------
Walks an entire target codebase (not just one file), runs every .py file
through the existing IndexingPipeline (parser -> chunker -> metadata ->
augmenter -> hashing), and saves ALL resulting chunks into one
data/chunks/chunks.jsonl.

This replaces the old `if __name__ == "__main__"` block in pipeline.py,
which only ever processed a single hardcoded "test.py" file -- that's why
the chunk count was stuck at 4 no matter what you searched.

Usage (run from the repo root):
    python scripts/build_index.py /path/to/target/codebase

If no path is given, defaults to indexing the repo's own src/ folder as a
reasonable stand-in so you have *something* bigger than 4 chunks to test
with immediately -- but you should point this at a real, separate
codebase (per the hackathon brief) before the actual demo.
"""

import sys
import traceback
from pathlib import Path

# Make sure src/ is importable regardless of where this is run from
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.indexing.pipeline import IndexingPipeline

# Directories to skip while walking the target codebase -- noise, not
# real application code worth indexing.
SKIP_DIRS = {
    "test", "tests", "__pycache__", "venv", ".venv", "env",
    "build", "dist", ".git", "node_modules", "site-packages",
    "migrations", "docs", "examples",
}


def find_python_files(root: Path):
    files = []
    for py_file in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        files.append(py_file)
    return files


def main():
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1]).resolve()
    else:
        target_dir = REPO_ROOT / "src"
        print(f"No target given -- defaulting to indexing this repo's own src/ folder: {target_dir}")
        print("(Point this at a real target codebase before your actual demo.)\n")

    if not target_dir.exists():
        print(f"ERROR: {target_dir} does not exist.")
        sys.exit(1)

    py_files = find_python_files(target_dir)
    print(f"Found {len(py_files)} Python files under {target_dir}\n")

    if not py_files:
        print("No .py files found -- nothing to index. Check the path you gave.")
        sys.exit(1)

    pipeline = IndexingPipeline()
    all_chunks = []
    failed = []

    for i, file_path in enumerate(py_files, 1):
        try:
            chunks = pipeline.process_file(file_path, language="python")
            all_chunks.extend(chunks)
            print(f"[{i}/{len(py_files)}] {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            failed.append((file_path, str(e)))
            print(f"[{i}/{len(py_files)}] {file_path.name}: FAILED ({e})")

    output_path = REPO_ROOT / "data" / "chunks" / "chunks.jsonl"
    pipeline.save_chunks(all_chunks, output_path)

    print(f"\n{'='*50}")
    print(f"Successfully indexed {len(all_chunks)} chunks from {len(py_files) - len(failed)} files.")
    print(f"Saved to: {output_path}")
    if failed:
        print(f"\n{len(failed)} file(s) failed to parse and were skipped:")
        for fp, err in failed[:10]:
            print(f"  - {fp.name}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
