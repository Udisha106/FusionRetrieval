"""
git_history.py
---------------
Reads git commit history for a repository and lets other modules check out
files at a specific commit, so we can re-chunk and re-index across versions.
"""

import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class CommitInfo:
    commit_hash: str
    message: str
    author: str
    timestamp: str


class GitHistory:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _run(self, args):
        result = subprocess.run(
            ["git", "-C", self.repo_path] + args,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def list_commits(self, branch="HEAD", limit=50):
        log_format = "%H|%s|%an|%aI"
        output = self._run(["log", branch, "-n" + str(limit), "--pretty=format:" + log_format, "--reverse"])
        commits = []
        for line in output.splitlines():
            if not line.strip():
                continue
            commit_hash, message, author, timestamp = line.split("|", 3)
            commits.append(CommitInfo(commit_hash, message, author, timestamp))
        return commits

    def list_files_at_commit(self, commit_hash):
        output = self._run(["ls-tree", "-r", "--name-only", commit_hash])
        return output.splitlines()

    def read_file_at_commit(self, commit_hash, file_path):
        return self._run(["show", commit_hash + ":" + file_path])

    def diff_files_between_commits(self, commit_a, commit_b):
        output = self._run(["diff", "--name-only", commit_a, commit_b])
        return output.splitlines()


if __name__ == "__main__":
    gh = GitHistory(repo_path=".")
    commits = gh.list_commits(limit=5)
    for c in commits:
        print(c.commit_hash[:8] + "  " + c.message + "  (" + c.author + ", " + c.timestamp + ")")
