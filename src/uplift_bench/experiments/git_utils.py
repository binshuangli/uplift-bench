"""Capture git provenance (commit hash, source-tree dirtiness) for result stamping.

Two subtleties, both surfaced by an external reproduction audit:

* The public repository's history is a single squashed release commit, so hashes stamped
  during development do not resolve on GitHub. Released parquets therefore carry a
  provenance note in PROVENANCE.md; verification of released results is by regeneration
  and comparison (``make repro FORCE=1 RESULTS_DIR=results_fresh``), not by checking out
  the stamped hash.
* ``is_dirty`` must consider SOURCE files only. Result parquets are committed, so a run
  that writes results/ makes the tree "dirty" by its own output -- which is not the
  contamination the flag exists to record. Output directories are excluded.
"""

from __future__ import annotations

import subprocess

# Directories whose changes are run OUTPUT, not source modification. A run that only
# touches these is still a clean-source run.
_OUTPUT_DIRS = (
    "results",
    "results_xgb",
    "results_m1ext",
    "results_ihdpval",
    "results_r4pred",
    "results_acic",
    "results_drB50",
    "results_notune",
    "results_auuctune",
    "results_revsynth",
    "paper",
    "paper_kdd",
)


def get_git_hash(short: bool = False) -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_dirty() -> bool:
    """True if SOURCE files (not run outputs) have uncommitted changes."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    for line in out.decode().splitlines():
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if not path.startswith(_OUTPUT_DIRS):
            return True
    return False
