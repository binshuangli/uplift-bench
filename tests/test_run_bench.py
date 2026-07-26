"""Regression tests for the orchestrator's failure semantics (external audits R24/R26).

run_one must (a) skip existing outputs without --force, (b) under --force quarantine
existing outputs to .stale and delete the quarantined copies only on verified success
(child returncode 0 AND a fresh parquet on disk), and (c) report error status when the
child exits 0 without writing anything -- so a failed forced rerun can never masquerade
as fresh results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_bench  # noqa: E402


class _FakeProc:
    def __init__(self, returncode: int):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = ""


def _fake_run(returncode: int, write_path: Path | None = None):
    def run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        if write_path is not None:
            write_path.write_bytes(b"parquet")
        return _FakeProc(returncode)

    return run


def _run_one(tmp_path: Path, force: bool) -> dict:
    return run_bench.run_one(
        job_id="ds__t_learner",
        dataset_group="synthetic",
        dataset_config="synthetic",
        model_name="t_learner",
        extra_overrides={},
        n_folds=3,
        n_seeds=3,
        n_samples=10,
        inner_folds=2,
        results_dir=tmp_path,
        data_dir=tmp_path,
        force=force,
    )


class TestOrchestratorSemantics:
    def test_existing_parquet_skipped_without_force(self, tmp_path, monkeypatch):
        (tmp_path / "ds__t_learner.parquet").write_bytes(b"old")

        def boom(*a, **k):  # the child must not be launched at all
            raise AssertionError("subprocess.run called despite existing parquet")

        monkeypatch.setattr(run_bench.subprocess, "run", boom)
        res = _run_one(tmp_path, force=False)
        assert res["status"] == "skipped_already_done"
        assert (tmp_path / "ds__t_learner.parquet").read_bytes() == b"old"

    def test_force_success_deletes_quarantine(self, tmp_path, monkeypatch):
        pq = tmp_path / "ds__t_learner.parquet"
        pq.write_bytes(b"old")
        (tmp_path / "ds__t_learner__summary.parquet").write_bytes(b"old")
        monkeypatch.setattr(run_bench.subprocess, "run", _fake_run(0, write_path=pq))
        res = _run_one(tmp_path, force=True)
        assert res["status"] == "ok"
        assert pq.read_bytes() == b"parquet"  # fresh output, not the old one
        assert not list(tmp_path.glob("*.stale"))

    def test_force_failure_leaves_stale_and_no_parquet(self, tmp_path, monkeypatch):
        pq = tmp_path / "ds__t_learner.parquet"
        pq.write_bytes(b"old")
        monkeypatch.setattr(run_bench.subprocess, "run", _fake_run(1))
        res = _run_one(tmp_path, force=True)
        assert res["status"] == "error"
        assert not pq.exists()  # no stale output masquerading as fresh
        stale = tmp_path / "ds__t_learner.parquet.stale"
        assert stale.exists() and stale.read_bytes() == b"old"

    def test_zero_exit_without_output_is_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_bench.subprocess, "run", _fake_run(0))
        res = _run_one(tmp_path, force=True)
        assert res["status"] == "error"
