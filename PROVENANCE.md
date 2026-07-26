# Result provenance

Every result parquet row is stamped with `git_hash` and `git_dirty` at run time. Two
facts an auditor needs to interpret those stamps:

## The stamped hashes do not resolve on GitHub — by design, not by accident

This public repository's history is a **single squashed release commit** per version tag:
the development history (which contains reviewer correspondence and draft manuscripts) is
private, and each release is published as one clean tree via `git commit-tree`. The
`git_hash` values inside the released parquets were stamped during development and refer
to that private history, so `git cat-file` against this repository will not find them.

What the stamps still give you: **within** the released results, rows sharing a hash were
produced by the same source state, and the run-to-run boundaries (main run vs. extension
runs vs. audit arms) are recoverable from the hash groups.

**How to verify the released results** — by regeneration, not by hash checkout:

```bash
make repro FORCE=1 RESULTS_DIR=results_fresh   # rerun the core benchmark into a new dir
python scripts/check_results_md.py             # released leaderboards vs released parquets
# then compare results_fresh/ against results/ (rank statistics should agree; fold-level
# values match up to library/BLAS nondeterminism documented in the paper's Appendix)
```

`FORCE=1` matters: the released parquets are committed, and without it the orchestrator
resumes (skips) every job whose output already exists.

## `git_dirty` in released parquets

Rows with `git_dirty=True` predate a fix to the dirtiness check: the old check counted
**newly written result files themselves** as working-tree changes, so any run that wrote
into a committed results directory was flagged dirty by its own output. The current
`git_utils.is_dirty()` considers source files only (output directories are excluded), so
the flag now records what it was always meant to: whether *source* was modified relative
to the stamped commit. The affected released rows were produced by unmodified source
plus in-flight outputs; the science is unaffected, and the flag is retained rather than
rewritten because editing released parquets would be worse provenance than documenting
them.
