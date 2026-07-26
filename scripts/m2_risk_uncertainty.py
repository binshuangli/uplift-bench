"""F2 policy-risk estimation noise + split-dependence sensitivity (R5, W2/W3/Q1/Q3).

Two reviewer concerns about the F2 intervals on Jobs:

(Q3 / W3) The IPW policy-risk estimand is high-variance (small experimental
subset, treated fraction ~0.09 overall), so regrets of ~0.026-0.028 must be
judged against the ESTIMATION noise of the risk itself, not only against the
bootstrap over splits. We quantify it directly: for every (realization, model)
we unit-level bootstrap the experimental-subset units within each evaluation
fold (holding predictions fixed) and report the SE of the per-realization
policy-risk estimate.

(Q1 / W2) The 10 Jobs "realizations" are train/test RE-SPLITS of the same
LaLonde+PSID observations (Shalit et al., 2017), not independent simulation
draws, so a cluster bootstrap that treats them as exchangeable may understate
uncertainty. We report three dependence-robust views of the F2 regret:
  (a) leave-one-split-out jackknife of the mean regret;
  (b) per-split sign counts (how many of the 10 splits show positive regret);
  (c) a design-effect threshold rho*: the largest average between-split
      correlation of per-split regrets under which the 95% interval still
      excludes zero, from SE_true = SE_iid * sqrt(1 + (n-1) rho). This makes
      the exchangeability assumption's role transparent instead of hidden.

Reads the Jobs per-unit prediction store (results_r4pred/predictions) plus the
per-fold parquets. Seeded. Emits results/tables/tab15_m2_uncertainty.tex.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.metrics.causal import jobs_policy_risk  # noqa: E402

SEED = 42
JOBS = [f"jobs_s{i}" for i in range(10)]
METRICS = [("qini", "Qini"), ("auuc", "AUUC"), ("uplift_at_k", "Uplift@k")]
SEEDS = (0, 1, 2)
N_UNIT_BOOT = 500


def _agg(df):
    return df.groupby("model").mean(numeric_only=True)


def unit_level_risk_ses(pred_dir="results_r4pred/predictions"):
    """SE of the per-realization policy risk from resampling experimental units."""
    rng = np.random.default_rng(SEED)
    rows = []
    for f in sorted(glob.glob(f"{pred_dir}/jobs_*__pred.parquet")):
        df = pd.read_parquet(f)
        ds, model = df.dataset.iloc[0], df.model.iloc[0]
        # per-realization risk = mean over (seed, fold) of fold risks; bootstrap
        # experimental units WITHIN each fold, holding predictions fixed.
        boots = np.zeros(N_UNIT_BOOT)
        base_folds = []
        fold_groups = [
            (
                g["pred"].to_numpy(),
                g["y"].to_numpy(),
                g["t"].to_numpy(),
                g["experimental"].to_numpy(),
            )
            for _, g in df.groupby(["seed_idx", "fold_idx"])
        ]
        for pred, y, t, e in fold_groups:
            base_folds.append(jobs_policy_risk(pred, y, t, e))
        for b in range(N_UNIT_BOOT):
            vals = []
            for pred, y, t, e in fold_groups:
                exp_idx = np.flatnonzero(e > 0)
                take = rng.choice(exp_idx, len(exp_idx), replace=True)
                keep = np.concatenate([np.flatnonzero(e <= 0), take])
                vals.append(jobs_policy_risk(pred[keep], y[keep], t[keep], e[keep]))
            boots[b] = np.mean(vals)
        rows.append(
            dict(
                dataset=ds,
                model=model,
                risk=float(np.mean(base_folds)),
                unit_se=float(boots.std(ddof=1)),
            )
        )
    return pd.DataFrame(rows)


def regret_dependence(results_dir="results"):
    """Per-split regrets (rotation-averaged) + jackknife + sign counts + rho*."""
    raw = pd.read_parquet(Path(results_dir) / "master_raw.parquet")
    raw = raw[(raw.status == "ok") & (raw.dataset.isin(JOBS)) & raw.jobs_policy_risk.notna()]
    out = {}
    for col, label in METRICS:
        per_split = []
        for ds in JOBS:
            d = raw[raw.dataset == ds]
            rot = []
            for ev_seed in SEEDS:
                sel = _agg(d[d.seed_idx != ev_seed])
                ev = _agg(d[d.seed_idx == ev_seed])
                c = sel.index.intersection(ev.index)
                if len(c) < 4:
                    continue
                rot.append(
                    float(
                        ev.loc[sel.loc[c, col].idxmax(), "jobs_policy_risk"]
                        - ev.loc[sel.loc[c, "jobs_policy_risk"].idxmin(), "jobs_policy_risk"]
                    )
                )
            if rot:
                per_split.append(np.mean(rot))
        v = np.asarray(per_split)
        n = len(v)
        mean = v.mean()
        se_iid = v.std(ddof=1) / np.sqrt(n)
        # (a) leave-one-out jackknife
        jk = np.array([np.delete(v, i).mean() for i in range(n)])
        jk_se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
        # (b) sign count
        pos = int((v > 0).sum())
        # (c) design-effect threshold: mean - 1.96*se_iid*sqrt(1+(n-1)rho) = 0
        z = mean / (1.96 * se_iid) if se_iid > 0 else np.inf
        rho_star = (z**2 - 1) / (n - 1) if np.isfinite(z) else np.inf
        rho_star = float(np.clip(rho_star, 0.0, 1.0)) if z > 1 else 0.0
        out[label] = dict(
            mean=mean, se_iid=se_iid, jk_se=jk_se, pos=pos, n=n, rho_star=rho_star, per_split=v
        )
    return out


def main(out_dir="results"):
    print("=== (Q3) Unit-level bootstrap SE of the IPW policy risk (per realization x model) ===")
    ses = unit_level_risk_ses()
    med = ses.unit_se.median()
    q10, q90 = ses.unit_se.quantile([0.1, 0.9])
    print(
        f"  {len(ses)} (realization, model) cells | median SE = {med:.4f} "
        f"| 10-90% = [{q10:.4f}, {q90:.4f}]"
    )
    by_model = ses.groupby("model").unit_se.median().sort_values()
    print(by_model.to_string())

    print("\n=== (Q1) Dependence-robust views of the F2 rotation regret ===")
    dep = regret_dependence()
    for label, d in dep.items():
        print(
            f"  {label:9s}: mean={d['mean']:+.4f} | iid SE={d['se_iid']:.4f} | "
            f"jackknife SE={d['jk_se']:.4f} | positive splits {d['pos']}/{d['n']} | "
            f"rho* = {d['rho_star']:.2f}"
        )
    print("  (rho* = max average between-split dependence at which the 95% CI still excludes 0)")

    q = dep["Qini"]
    macros = [
        r"% auto-generated by scripts/m2_risk_uncertainty.py (R5 W2/W3)",
        f"\\newcommand{{\\riskUnitSeMed}}{{{med:.3f}}}",
        f"\\newcommand{{\\riskUnitSeLo}}{{{q10:.3f}}}",
        f"\\newcommand{{\\riskUnitSeHi}}{{{q90:.3f}}}",
        f"\\newcommand{{\\regJkSeQini}}{{{q['jk_se']:.4f}}}",
        f"\\newcommand{{\\regPosQini}}{{{q['pos']}}}",
        f"\\newcommand{{\\regRhoStarQini}}{{{q['rho_star']:.2f}}}",
        f"\\newcommand{{\\regRhoStarAuuc}}{{{dep['AUUC']['rho_star']:.2f}}}",
        f"\\newcommand{{\\regRhoStarUplift}}{{{dep['Uplift@k']['rho_star']:.2f}}}",
        f"\\newcommand{{\\regPosAuuc}}{{{dep['AUUC']['pos']}}}",
        f"\\newcommand{{\\regPosUplift}}{{{dep['Uplift@k']['pos']}}}",
    ]
    outp = Path(out_dir) / "tables" / "tab15_m2_uncertainty.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")
    print(f"Saved: {outp}")


if __name__ == "__main__":
    main()
