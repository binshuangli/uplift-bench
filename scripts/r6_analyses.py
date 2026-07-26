"""R6 analyses B1 / B2 / B4 (consolidated-revision guide; no model re-runs).

B1  Policy-value-reference selectors across budgets. The R4 budget grid used the
    sign-threshold policy risk as the reference at every budget; this analysis
    re-runs the cross-repeat rotation with the reference selector being
    POLICY VALUE AT THE SAME BUDGET (V_k), for k in {0.1,0.2,0.3,0.5}:
    select by Qini / AUUC / uplift-at-k / V_k on the selection repetitions,
    evaluate all on held-out V_k, and compare each with random selection.
    Answers: is F2 specific to the sign-threshold objective?

B2  IHDP-only (leave-Synthetic-out) summary on the 10 main splits with the FULL
    12-estimator panel at B=10 — the only Synthetic-free instance of the
    headline configuration: mean Qini-(-sqrt PEHE) rho with bootstrap CI, and
    the paired AUUC-Qini gap Delta with CI.

B4  Variance-reduced Qini, best case: Qini computed on ORACLE-baseline-adjusted
    outcomes y - mu0(x) (available on IHDP), on the identical stored
    predictions. If even oracle control-outcome adjustment (the ceiling of any
    variance-reduction scheme, cf. the baseline-adjusted-score corollary) does
    not restore alignment with -sqrt(PEHE), estimation variance of the Qini
    functional is not the driver of F1.

Reads results/master_raw.parquet, results_r4pred/fold_metrics_cache.parquet,
and results_r4pred/predictions/. Seeded. Emits results/tables/tab16_r6.tex.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.data.registry import load_dataset  # noqa: E402
from uplift_bench.metrics.ranking import qini_coefficient  # noqa: E402

SEED = 42
KGRID = (0.1, 0.2, 0.3, 0.5)
SEEDS = (0, 1, 2)
JOBS = [f"jobs_s{i}" for i in range(10)]
IHDP_MAIN = [f"ihdp_s{i}" for i in range(10)]


def _boot_ci(v, n=10000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"),) * 3
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def b1_value_reference_selectors(macros):
    cache = pd.read_parquet("results_r4pred/fold_metrics_cache.parquet")
    # The prediction cache computes AUUC itself (r4_prediction_analyses.py), so use it
    # directly; merging a second `auuc` column from master_raw would collide into
    # auuc_x/auuc_y and break every selector keyed on "auuc".
    if "auuc" not in cache.columns:
        raw = pd.read_parquet("results/master_raw.parquet")
        auuc = raw[["dataset", "model", "seed_idx", "fold_idx", "auuc"]]
        cache = cache.merge(auuc, on=["dataset", "model", "seed_idx", "fold_idx"], how="left")
    df = cache[cache.dataset.isin(JOBS)]

    print("=== B1: rotation with the SAME-BUDGET policy value V_k as reference ===")
    print("    (select on 2 seeds, evaluate held-out V_k; regret vs V_k-selected reference;")
    print("     gain vs random; cluster bootstrap over the 10 realizations)")
    for k in KGRID:
        vk = f"pv_at_{k}"
        selectors = [("qini_canonical", "Qini"), ("auuc", "AUUC"),
                     (f"uplift_at_{k}", f"Up@{k}"), (vk, f"V@{k} (ref self)")]
        regs = {lab: [] for _, lab in selectors}
        gains = {lab: [] for _, lab in selectors}
        for ds in JOBS:
            d = df[df.dataset == ds]
            rot_r = {lab: [] for _, lab in selectors}
            rot_g = {lab: [] for _, lab in selectors}
            for ev_seed in SEEDS:
                sel = d[d.seed_idx != ev_seed].groupby("model").mean(numeric_only=True)
                ev = d[d.seed_idx == ev_seed].groupby("model").mean(numeric_only=True)
                common = sel.index.intersection(ev.index)
                if len(common) < 4:
                    continue
                v_ev = ev.loc[common, vk]
                ref = sel.loc[common, vk].idxmax()          # V_k-selected reference
                rand = float(v_ev.mean())
                for col, lab in selectors:
                    m_star = sel.loc[common, col].idxmax()
                    rot_r[lab].append(float(v_ev[ref] - v_ev[m_star]))   # value regret
                    rot_g[lab].append(float(v_ev[m_star] - rand))        # gain vs random
            for _, lab in selectors:
                if rot_r[lab]:
                    regs[lab].append(np.mean(rot_r[lab]))
                    gains[lab].append(np.mean(rot_g[lab]))
        line = [f"  k={k}:"]
        for _, lab in selectors:
            m, lo, hi = _boot_ci(regs[lab])
            gm, glo, ghi = _boot_ci(gains[lab])
            beats = "+" if glo > 0 else ("-" if ghi < 0 else "0")
            line.append(f"{lab}: reg={m:+.4f}[{lo:+.4f},{hi:+.4f}] "
                        f"randgain={gm:+.4f}[{glo:+.4f},{ghi:+.4f}]({beats})")
        print("\n    ".join(line))
        if k == 0.3:
            for (_, lab), mac in zip(selectors, ("QiniVref", "AuucVref", "UpliftVref", "SelfVref")):
                m, lo, hi = _boot_ci(regs[lab])
                gm, glo, ghi = _boot_ci(gains[lab])
                macros += [
                    f"\\newcommand{{\\bOne{mac}Reg}}{{{m:+.3f}}}",
                    f"\\newcommand{{\\bOne{mac}RegLo}}{{{lo:+.3f}}}",
                    f"\\newcommand{{\\bOne{mac}RegHi}}{{{hi:+.3f}}}",
                    f"\\newcommand{{\\bOne{mac}Gain}}{{{gm:+.3f}}}",
                    f"\\newcommand{{\\bOne{mac}GainLo}}{{{glo:+.3f}}}",
                    f"\\newcommand{{\\bOne{mac}GainHi}}{{{ghi:+.3f}}}",
                ]


def b2_ihdp_only(macros):
    raw = pd.read_parquet("results/master_raw.parquet")
    raw = raw[(raw.status == "ok") & raw.pehe.notna() & raw.dataset.isin(IHDP_MAIN)]
    rq, ra = [], []
    for ds in IHDP_MAIN:
        g = raw[raw.dataset == ds].groupby("model")[["qini", "auuc", "pehe"]].mean().dropna()
        if len(g) >= 4:
            rq.append(spearmanr(g.qini, -g.pehe).correlation)
            ra.append(spearmanr(g.auuc, -g.pehe).correlation)
    rq, ra = np.asarray(rq), np.asarray(ra)
    d = ra - rq
    mq, qlo, qhi = _boot_ci(rq)
    ma, alo, ahi = _boot_ci(ra)
    md, dlo, dhi = _boot_ci(d)
    print("\n=== B2: IHDP-only (10 main splits, full 12-estimator panel, B=10) ===")
    print(f"  Qini  vs -PEHE: {mq:+.3f} [{qlo:+.3f},{qhi:+.3f}]")
    print(f"  AUUC  vs -PEHE: {ma:+.3f} [{alo:+.3f},{ahi:+.3f}]")
    print(f"  paired Delta (AUUC-Qini): {md:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
          f" positive {int((d > 0).sum())}/{len(d)}")
    macros += [
        f"\\newcommand{{\\bTwoQini}}{{{mq:+.2f}}}",
        f"\\newcommand{{\\bTwoQiniLo}}{{{qlo:+.2f}}}",
        f"\\newcommand{{\\bTwoQiniHi}}{{{qhi:+.2f}}}",
        f"\\newcommand{{\\bTwoAuuc}}{{{ma:+.2f}}}",
        f"\\newcommand{{\\bTwoDelta}}{{{md:+.2f}}}",
        f"\\newcommand{{\\bTwoDeltaLo}}{{{dlo:+.2f}}}",
        f"\\newcommand{{\\bTwoDeltaHi}}{{{dhi:+.2f}}}",
    ]


def b4_oracle_adjusted_qini(macros):
    print("\n=== B4: oracle-baseline-adjusted (variance-reduced ceiling) Qini on IHDP ===")
    files = sorted(glob.glob("results_r4pred/predictions/ihdp_s*__pred.parquet"))
    mu0_cache: dict[str, np.ndarray] = {}
    rows = []
    for f in files:
        df = pd.read_parquet(f)
        ds = df.dataset.iloc[0]
        if ds not in mu0_cache:
            split = int(ds.split("_s")[1])
            mu0_cache[ds] = load_dataset("ihdp", split_idx=split).ite["mu0"].to_numpy()
        mu0 = mu0_cache[ds]
        for (model, seed, fold), g in df.groupby(["model", "seed_idx", "fold_idx"]):
            y_adj = g["y"].to_numpy() - mu0[g["unit_id"].to_numpy()]
            rows.append(dict(
                dataset=ds, model=model, seed_idx=seed, fold_idx=fold,
                qini_adj=qini_coefficient(g["pred"].to_numpy(), g["t"].to_numpy(),
                                          y_adj, normalize=False),
                qini_raw=qini_coefficient(g["pred"].to_numpy(), g["t"].to_numpy(),
                                          g["y"].to_numpy(), normalize=False),
                pehe=float(np.sqrt(np.mean((g["pred"].to_numpy()
                                            - g["ite"].to_numpy()) ** 2))),
            ))
    fold = pd.DataFrame(rows)
    r_adj, r_raw = [], []
    for ds, g in fold.groupby("dataset"):
        gm = g.groupby("model")[["qini_adj", "qini_raw", "pehe"]].mean().dropna()
        if len(gm) >= 4:
            r_adj.append(spearmanr(gm.qini_adj, -gm.pehe).correlation)
            r_raw.append(spearmanr(gm.qini_raw, -gm.pehe).correlation)
    ma, alo, ahi = _boot_ci(r_adj)
    mr, rlo, rhi = _boot_ci(r_raw)
    print(f"  raw Qini            vs -PEHE: {mr:+.3f} [{rlo:+.3f},{rhi:+.3f}] (n={len(r_raw)})")
    print(f"  oracle-adjusted Qini vs -PEHE: {ma:+.3f} [{alo:+.3f},{ahi:+.3f}]")
    macros += [
        f"\\newcommand{{\\bFourRaw}}{{{mr:+.2f}}}".replace("-0.00","+0.00"),
        f"\\newcommand{{\\bFourAdj}}{{{ma:+.2f}}}",
        f"\\newcommand{{\\bFourAdjLo}}{{{alo:+.2f}}}",
        f"\\newcommand{{\\bFourAdjHi}}{{{ahi:+.2f}}}",
        f"\\newcommand{{\\bFourN}}{{{len(r_adj)}}}",
    ]


def main(out_dir="results"):
    macros = ["% auto-generated by scripts/r6_analyses.py (B1/B2/B4)"]
    b1_value_reference_selectors(macros)
    b2_ihdp_only(macros)
    b4_oracle_adjusted_qini(macros)
    outp = Path(out_dir) / "tables" / "tab16_r6.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")
    print(f"\nSaved: {outp}")


if __name__ == "__main__":
    main()
