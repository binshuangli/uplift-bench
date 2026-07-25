"""Capture the current git commit hash for result provenance."""

from __future__ import annotations

import subprocess


def get_git_hash(short: bool = False) -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return bool(out.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
