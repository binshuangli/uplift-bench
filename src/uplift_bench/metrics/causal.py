"""Ground-truth CATE metrics (IHDP / Jobs).

These require known potential outcomes or an experimental benchmark subset.
"""

from __future__ import annotations

import numpy as np


def pehe(cate_pred, cate_true) -> float:
    """Precision in Estimation of Heterogeneous Effects.

    sqrt(E[(tau_pred - tau_true)^2])  — a.k.a. root-MSE of the CATE.
    """
    cate_pred = np.asarray(cate_pred, dtype=float)
    cate_true = np.asarray(cate_true, dtype=float)
    return float(np.sqrt(np.mean((cate_pred - cate_true) ** 2)))


def epsilon_ate(cate_pred, cate_true) -> float:
    """Absolute error in the estimated ATE.

    |mean(tau_pred) - mean(tau_true)|
    """
    cate_pred = np.asarray(cate_pred, dtype=float)
    cate_true = np.asarray(cate_true, dtype=float)
    return float(np.abs(cate_pred.mean() - cate_true.mean()))


def jobs_policy_risk(
    uplift_score,
    outcome,
    treatment,
    experimental_flag,
    threshold: float = 0.0,
) -> float:
    """Policy risk on the Jobs experimental subset (LaLonde RCT units).

    Follows Shalit et al. (2017): policy risk = 1 - E[Y(π(x)) | experimental].

    The policy π assigns treatment when predicted uplift >= threshold.
    We evaluate only on the experimental (randomised) subset where
    the treated and control potential outcomes can be compared fairly.

    Parameters
    ----------
    uplift_score      : predicted uplift for every unit
    outcome           : observed factual outcome
    treatment         : observed treatment assignment
    experimental_flag : 1 if unit is in the LaLonde experimental group, else 0
    threshold         : treat unit if uplift_score >= threshold (default 0)
    """
    uplift_score = np.asarray(uplift_score, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    experimental_flag = np.asarray(experimental_flag, dtype=float)

    mask = experimental_flag == 1
    if mask.sum() == 0:
        raise ValueError("No experimental units found (experimental_flag all 0).")

    score_e = uplift_score[mask]
    y_e = outcome[mask]
    t_e = treatment[mask]
    policy_e = (score_e >= threshold).astype(float)

    # IPW estimate of E[Y(π(x))]
    # For experimental subset: P(T=1) ≈ empirical fraction
    prop = float(t_e.mean())
    if prop <= 0 or prop >= 1:
        raise ValueError(f"Experimental propensity out of (0,1): {prop}")

    with np.errstate(invalid="ignore", divide="ignore"):
        y_policy_ipw = np.where(
            (policy_e == 1) & (t_e == 1),
            y_e / prop,
            np.where(
                (policy_e == 0) & (t_e == 0),
                y_e / (1 - prop),
                0.0,
            ),
        )
    e_y_pi = float(y_policy_ipw.mean())
    # Risk = 1 - value (lower is better; Shalit convention)
    return float(1.0 - e_y_pi)
