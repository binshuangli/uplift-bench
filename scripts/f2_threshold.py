"""F2 calibrated-threshold experiment (reviewer Q2 follow-up).

The rank-invariance proposition says ranking metrics are blind to the score
LEVEL, which is exactly what the sign-threshold policy pi(x)=1[tau_hat(x)>=0]
depends on. If that level-blindness is the driver of F2, then giving the
metric-selected model a threshold CALIBRATED on the selection folds (rather than
the fixed 0) should reduce its held-out policy risk toward the reference --- i.e.
"ranking metric + calibrated threshold" should close much of the gap.

Design (same cross-repeat rotation as m2_selection_signal / m2_regret):
  For each Jobs realization and each held-out seed (the other two seeds = selection):
    - pool each seed's 3 out-of-fold predictions (each seed covers all units once);
    - for every model, calibrate a threshold on the SELECTION pool:
        thr*_m = argmin_thr policy_risk(model, thr) over a score-quantile grid;
    - metric winner  m* = argmax Qini on the selection pool;
    - reference r0 = argmin policy_risk(.,thr=0) on selection (sign-threshold ref, = F2);
      reference rc = argmin_m policy_risk(m, thr*_m) on selection (calibrated ref);
    - evaluate on the held-out seed:
        regret_sign  = risk(m*, thr=0)      - risk(r0, thr=0)         [current F2]
        regret_calib = risk(m*, thr*_{m*})  - risk(rc, thr*_{rc})     [calibrated]
Average the 3 rotations within a realization, cluster-bootstrap over the 10.
Also report the metric-winner's own risk drop from calibrating its threshold.

Reads results_r4pred/predictions/jobs_*__pred.parquet. Seeded.
Emits results/tables/tab19_threshold.tex + prints a summary.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uplift_bench.metrics.causal import jobs_policy_risk  # noqa: E402
from uplift_bench.metrics.ranking import qini_coefficient  # noqa: E402

SEED = 42
SEEDS = (0, 1, 2)
N_GRID = 41  # threshold candidates = score quantiles


def _boot_ci(v, n=10000):
    rng = np.random.default_rng(SEED)
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _seed_pool(df, seed):
    """Pool a seed's out-of-fold predictions (its 3 folds cover all units once)."""
    g = df[df.seed_idx == seed]
    return (g["pred"].to_numpy(), g["y"].to_numpy(), g["t"].to_numpy(),
            g["experimental"].to_numpy())


def _best_threshold(pred, y, t, e):
    grid = np.quantile(pred, np.linspace(0.02, 0.98, N_GRID))
    risks = [jobs_policy_risk(pred, y, t, e, threshold=thr) for thr in grid]
    return float(grid[int(np.argmin(risks))])


def main(out_dir="results"):
    files = sorted(glob.glob("results_r4pred/predictions/jobs_*__pred.parquet"))
    # group prediction frames by realization
    by_ds: dict[str, dict[str, pd.DataFrame]] = {}
    for f in files:
        df = pd.read_parquet(f)
        by_ds.setdefault(df.dataset.iloc[0], {})[df.model.iloc[0]] = df

    reg_sign, reg_calib, winner_drop = [], [], []
    for ds, models in by_ds.items():
        rot_sign, rot_calib, rot_drop = [], [], []
        for ev in SEEDS:
            sels = [s for s in SEEDS if s != ev]
            # per-model selection pool (concatenate the two selection seeds) + eval pool
            sel_pool, ev_pool, thr_star, qini_sel, risk0_sel = {}, {}, {}, {}, {}
            for m, df in models.items():
                ps = [ _seed_pool(df, s) for s in sels ]
                pred_s = np.concatenate([p[0] for p in ps])
                y_s = np.concatenate([p[1] for p in ps])
                t_s = np.concatenate([p[2] for p in ps])
                e_s = np.concatenate([p[3] for p in ps])
                sel_pool[m] = (pred_s, y_s, t_s, e_s)
                ev_pool[m] = _seed_pool(df, ev)
                if e_s.sum() < 10:
                    continue
                thr_star[m] = _best_threshold(pred_s, y_s, t_s, e_s)
                qini_sel[m] = qini_coefficient(pred_s, t_s, y_s, normalize=False)
                risk0_sel[m] = jobs_policy_risk(pred_s, y_s, t_s, e_s, threshold=0.0)
            common = [m for m in models if m in thr_star]
            if len(common) < 4:
                continue
            # winners on selection
            m_star = max(common, key=lambda m: qini_sel[m])                 # Qini winner
            ref0 = min(common, key=lambda m: risk0_sel[m])                 # sign-threshold ref
            ref_c = min(common, key=lambda m: jobs_policy_risk(*sel_pool[m],  # calibrated ref
                                                              threshold=thr_star[m]))
            # evaluate on held-out seed
            def ev_risk(m, thr):
                return jobs_policy_risk(*ev_pool[m], threshold=thr)
            rot_sign.append(ev_risk(m_star, 0.0) - ev_risk(ref0, 0.0))
            rot_calib.append(ev_risk(m_star, thr_star[m_star]) - ev_risk(ref_c, thr_star[ref_c]))
            rot_drop.append(ev_risk(m_star, 0.0) - ev_risk(m_star, thr_star[m_star]))
        if rot_sign:
            reg_sign.append(np.mean(rot_sign))
            reg_calib.append(np.mean(rot_calib))
            winner_drop.append(np.mean(rot_drop))

    s_m, s_lo, s_hi = _boot_ci(reg_sign)
    c_m, c_lo, c_hi = _boot_ci(reg_calib)
    d_m, d_lo, d_hi = _boot_ci(winner_drop)
    closed = 100 * (1 - c_m / s_m) if s_m != 0 else float("nan")

    print("=== F2 calibrated-threshold experiment (Qini selector, 10 Jobs realizations) ===")
    print(f"  sign-threshold regret (F2):   {s_m:+.4f} [{s_lo:+.4f},{s_hi:+.4f}]")
    print(f"  calibrated-threshold regret:  {c_m:+.4f} [{c_lo:+.4f},{c_hi:+.4f}]")
    print(f"  gap closed by calibration:    {closed:.0f}%")
    print(f"  metric-winner risk drop from calibrating its own threshold: "
          f"{d_m:+.4f} [{d_lo:+.4f},{d_hi:+.4f}]")

    macros = [
        r"% auto-generated by scripts/f2_threshold.py (Q2)",
        f"\\newcommand{{\\thrRegSign}}{{{s_m:+.4f}}}",
        f"\\newcommand{{\\thrRegSignLo}}{{{s_lo:+.4f}}}",
        f"\\newcommand{{\\thrRegSignHi}}{{{s_hi:+.4f}}}",
        f"\\newcommand{{\\thrRegCalib}}{{{c_m:+.4f}}}",
        f"\\newcommand{{\\thrRegCalibLo}}{{{c_lo:+.4f}}}",
        f"\\newcommand{{\\thrRegCalibHi}}{{{c_hi:+.4f}}}",
        f"\\newcommand{{\\thrClosed}}{{{closed:.0f}}}",
        f"\\newcommand{{\\thrWinnerDrop}}{{{d_m:+.4f}}}",
        f"\\newcommand{{\\thrWinnerDropLo}}{{{d_lo:+.4f}}}",
        f"\\newcommand{{\\thrWinnerDropHi}}{{{d_hi:+.4f}}}",
    ]
    outp = Path(out_dir) / "tables" / "tab19_threshold.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(macros) + "\n")
    print(f"Saved: {outp}")


if __name__ == "__main__":
    main()
