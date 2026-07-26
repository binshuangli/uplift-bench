"""Uplift calibration: reliability curve and calibration error.

Uplift calibration checks whether predicted uplift scores match the
observed incremental outcome rates across score buckets.

A well-calibrated uplift model satisfies:
    E[Y | T=1, score in bucket b] - E[Y | T=0, score in bucket b]  ≈  mean(score in bucket b)
"""

from __future__ import annotations

import numpy as np


def uplift_reliability_curve(
    uplift_score,
    treatment,
    outcome,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict:
    """Compute uplift reliability (calibration) curve.

    Parameters
    ----------
    uplift_score : predicted uplift, shape (n,)
    treatment    : binary {0,1}, shape (n,)
    outcome      : binary or continuous, shape (n,)
    n_bins       : number of calibration bins
    strategy     : 'quantile' (equal-count) or 'uniform' (equal-width)

    Returns
    -------
    dict with keys:
        bin_centers     : midpoint of each bin (mean predicted uplift)
        observed_uplift : observed uplift (diff-in-means) per bin
        bin_counts      : total units per bin
        bin_treated     : treated units per bin
        bin_control     : control units per bin
    """
    uplift_score = np.asarray(uplift_score, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    outcome = np.asarray(outcome, dtype=float)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.unique(np.quantile(uplift_score, quantiles))
    elif strategy == "uniform":
        lo, hi = uplift_score.min(), uplift_score.max()
        bins = np.linspace(lo, hi, n_bins + 1) if hi > lo else np.array([lo, lo + 1e-9])
    else:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")

    # Degenerate case: all scores identical → single bin covering everything
    if len(bins) < 2:
        bins = np.array([uplift_score.min() - 1e-9, uplift_score.max() + 1e-9])

    bin_centers = []
    observed_uplift = []
    bin_counts = []
    bin_treated = []
    bin_control = []

    bin_ids = np.digitize(uplift_score, bins[1:-1])  # 0-indexed bin assignment

    for b in range(len(bins) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        s_b = uplift_score[mask]
        t_b = treatment[mask]
        y_b = outcome[mask]
        n_t = t_b.sum()
        n_c = (1 - t_b).sum()

        bin_centers.append(float(s_b.mean()))
        bin_counts.append(int(mask.sum()))
        bin_treated.append(int(n_t))
        bin_control.append(int(n_c))

        if n_t > 0 and n_c > 0:
            obs = float(y_b[t_b == 1].mean() - y_b[t_b == 0].mean())
        else:
            obs = float("nan")
        observed_uplift.append(obs)

    return {
        "bin_centers": np.array(bin_centers),
        "observed_uplift": np.array(observed_uplift),
        "bin_counts": np.array(bin_counts),
        "bin_treated": np.array(bin_treated),
        "bin_control": np.array(bin_control),
    }


def uplift_calibration_error(
    uplift_score,
    treatment,
    outcome,
    n_bins: int = 10,
    strategy: str = "quantile",
    weighted: bool = True,
) -> float:
    """Expected Calibration Error (ECE) for uplift models.

    ECE = sum_b [ (n_b / n) * |predicted_uplift_b - observed_uplift_b| ]

    NaN bins (insufficient treated or control) are excluded from the sum.

    Parameters
    ----------
    weighted : if True, weight by bin size (ECE); if False, unweighted mean.
    """
    curve = uplift_reliability_curve(
        uplift_score, treatment, outcome, n_bins=n_bins, strategy=strategy
    )
    predicted = curve["bin_centers"]
    observed = curve["observed_uplift"]
    counts = curve["bin_counts"]

    valid = ~np.isnan(observed)
    if valid.sum() == 0:
        return float("nan")

    predicted = predicted[valid]
    observed = observed[valid]
    counts = counts[valid]

    abs_err = np.abs(predicted - observed)
    if weighted:
        return float(np.sum(counts * abs_err) / counts.sum())
    return float(abs_err.mean())
