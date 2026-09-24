import os
from pathlib import Path
import subprocess
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_provenance() -> Dict[str, str]:
    """Git commit and dirty flag of the training code.

    Hosts without a .git directory (euler gets a copied tree) can pass the commit via the
    CTLOG_GIT_COMMIT environment variable.

    Returns:
        Dict[str, str]: git_commit and git_dirty ("true", "false" or "unknown").
    """
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = "true" if _git("status", "--porcelain", "--untracked-files=no") else "false"
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("CTLOG_GIT_COMMIT", "unknown")
        dirty = "unknown"
    return {"git_commit": commit, "git_dirty": dirty}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
