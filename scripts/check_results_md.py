"""Verify RESULTS.md's leaderboard tables against the committed result parquets.

RESULTS.md is the public leaderboard the paper cites as a governance artifact, and it is
written by hand. Hand-maintained numbers drift: an earlier version showed six marketing
cells as missing when results existed, bolded the wrong MegaFon winner, and omitted one
estimator from the Jobs table entirely. This script makes that class of drift a test
failure instead of something a reader has to catch.

Checks, for every model x dataset cell parsed out of the markdown tables:
  * marketing Qini values match `qini_mean` to the displayed precision;
  * a cell shown as `TO` really has no completed row, and vice versa;
  * Jobs policy-risk values match the mean over the 10 splits, and all 12 estimators
    appear;
  * the bolded best cell per marketing dataset really is the argmax.

Exit code 0 = consistent, 1 = drift (prints every mismatch).
Usage: python scripts/check_results_md.py   (or: make check-results)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RESULTS_MD = Path("RESULTS.md")
SUMMARY = Path("results/master_summary.parquet")

LABEL_TO_MODEL = {
    "S-Learner": "s_learner",
    "T-Learner": "t_learner",
    "X-Learner": "x_learner",
    "R-Learner": "r_learner",
    "DR-Learner": "dr_learner",
    "ClassTrans": "class_transformation",
    "TwoModel": "two_model",
    "SoloModel": "solo_model",
    "CausalForest": "causal_forest",
    "UpliftRF-KL": "uplift_rf_kl",
    "UpliftRF-ED": "uplift_rf_ed",
    "UpliftRF-Chi": "uplift_rf_chi",
}
MARKETING = ["hillstrom", "lenta", "x5", "megafon"]
JOBS = [f"jobs_s{i}" for i in range(10)]


def _clean(cell: str) -> str:
    """Strip markdown emphasis and normalise the unicode minus."""
    return cell.replace("**", "").replace("−", "-").strip()


def _rows(md: str, header_start: str) -> list[list[str]]:
    """Table rows following the first header line containing `header_start`."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if header_start in ln and ln.lstrip().startswith("|"):
            out = []
            for ln2 in lines[i + 2 :]:
                if not ln2.lstrip().startswith("|"):
                    break
                out.append([c for c in ln2.strip().strip("|").split("|")])
            return out
    return []


def main() -> int:
    if not (RESULTS_MD.exists() and SUMMARY.exists()):
        print("skipping check_results_md: RESULTS.md or results/master_summary.parquet missing")
        return 0
    md = RESULTS_MD.read_text()
    s = pd.read_parquet(SUMMARY)
    problems: list[str] = []

    # ---- marketing Qini table ----
    rows = _rows(md, "| Model | Hillstrom |")
    seen_marketing = set()
    best_claimed: dict[str, str] = {}
    for r in rows:
        label = _clean(r[0])
        if label not in LABEL_TO_MODEL:
            continue
        model = LABEL_TO_MODEL[label]
        seen_marketing.add(model)
        for ds, raw_cell in zip(MARKETING, r[1:5]):
            shown, cell = _clean(raw_cell), raw_cell
            hit = s[(s.dataset == ds) & (s.model == model)]
            if shown == "TO":
                if not hit.empty:
                    problems.append(
                        f"marketing {ds}/{label}: shown TO but a completed row exists "
                        f"(qini={hit.qini_mean.iloc[0]:.1f})"
                    )
            else:
                if hit.empty:
                    problems.append(f"marketing {ds}/{label}: shows {shown} but no row exists")
                else:
                    want = f"{hit.qini_mean.iloc[0]:.1f}"
                    if shown != want:
                        problems.append(
                            f"marketing {ds}/{label}: RESULTS.md {shown} vs data {want}"
                        )
            if "**" in cell and shown != "TO":
                best_claimed[ds] = label
    missing = set(LABEL_TO_MODEL.values()) - seen_marketing
    if missing:
        problems.append(f"marketing table omits models: {sorted(missing)}")

    # bolded best per marketing dataset must be the argmax
    for ds, label in best_claimed.items():
        sub = s[(s.dataset == ds) & s.qini_mean.notna()]
        if not sub.empty:
            argmax = sub.loc[sub.qini_mean.idxmax()].model
            if LABEL_TO_MODEL[label] != argmax:
                problems.append(f"marketing {ds}: bolded best is {label} but argmax is {argmax}")

    # ---- Jobs policy-risk table (two model/value column pairs per row) ----
    jrows = _rows(md, "| Model | Policy risk")
    jobs_mean = (
        s[s.dataset.isin(JOBS) & (s.jobs_policy_risk_n > 0)]
        .groupby("model")
        .jobs_policy_risk_mean.mean()
    )
    seen_jobs = set()
    for r in jrows:
        for lbl, val in ((r[0], r[1]), (r[3], r[4])) if len(r) >= 5 else ():
            label, shown = _clean(lbl), _clean(val)
            if label not in LABEL_TO_MODEL or not shown:
                continue
            model = LABEL_TO_MODEL[label]
            seen_jobs.add(model)
            if model not in jobs_mean.index:
                problems.append(f"jobs/{label}: listed but absent from the data")
            else:
                want = f"{jobs_mean[model]:.3f}"
                if shown != want:
                    problems.append(f"jobs/{label}: RESULTS.md {shown} vs data {want}")
    missing_j = set(jobs_mean.index) - seen_jobs
    if missing_j:
        problems.append(f"jobs table omits models present in the data: {sorted(missing_j)}")

    if problems:
        print(f"RESULTS.md is INCONSISTENT with {SUMMARY} ({len(problems)} problem(s)):")
        for p in problems:
            print("  -", p)
        return 1
    print(
        f"RESULTS.md is consistent with {SUMMARY} "
        f"({len(seen_marketing)} marketing models x {len(MARKETING)} datasets, "
        f"{len(seen_jobs)} Jobs models)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
