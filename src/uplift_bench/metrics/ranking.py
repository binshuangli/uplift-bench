"""Ranking / policy metrics for uplift evaluation.

All functions follow the signature:
    metric(uplift_score, treatment, outcome, **kwargs) -> float

where
    uplift_score : array-like (n,)  — predicted uplift / CATE
    treatment    : array-like (n,)  — binary {0, 1}
    outcome      : array-like (n,)  — binary or continuous
"""

from __future__ import annotations

import numpy as np


def _to_arrays(*arrs):
    return tuple(np.asarray(a, dtype=float) for a in arrs)


# ---------------------------------------------------------------------------
# Qini coefficient
# ---------------------------------------------------------------------------


def _qini_curve(uplift_score, treatment, outcome):
    """
    Return (fractions, qini_values, perfect_qini_values).

    The Qini curve plots cumulative incremental gain vs fraction of population
    targeted, sorting by descending predicted uplift.

    Incremental gain at position k =
        (sum_k(y | t=1) / n_t) - (sum_k(y | t=0) / n_c)  * n_t_k

    Following Radcliffe (2007) and the scikit-uplift convention.
    """
    uplift_score, treatment, outcome = _to_arrays(uplift_score, treatment, outcome)
    n = len(uplift_score)
    order = np.argsort(-uplift_score)  # descending
    t = treatment[order]
    y = outcome[order]

    n_t_total = treatment.sum()
    n_c_total = (1 - treatment).sum()
    if n_t_total == 0 or n_c_total == 0:
        raise ValueError("Need both treated and control units.")

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))

    # Avoid division by zero in sparse prefixes
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.where(
            cum_t > 0,
            cum_yt - cum_yc * (cum_t / np.where(cum_c > 0, cum_c, 1)),
            0.0,
        )

    fractions = np.arange(1, n + 1) / n
    # Prepend (0, 0) for trapezoid integration
    fractions = np.concatenate([[0.0], fractions])
    gain = np.concatenate([[0.0], gain])

    # Perfect curve: sort treated positives first, then treated negatives, then controls
    # (oracle ranking that maximises the Qini)
    n_treated_pos = int((treatment * outcome).sum())
    perfect_gain = np.zeros(n + 1)
    for k in range(1, n + 1):
        tp = min(k, n_treated_pos)
        tn = max(0, min(k - tp, int(n_t_total) - n_treated_pos))
        cp = max(0, k - tp - tn)
        if n_t_total > 0 and n_c_total > 0:
            perfect_gain[k] = tp - cp * (n_t_total / n_c_total)
        else:
            perfect_gain[k] = 0.0

    return fractions, gain, perfect_gain


def qini_coefficient(
    uplift_score,
    treatment,
    outcome,
    normalize: bool = True,
) -> float:
    """Qini coefficient (area between model curve and random baseline).

    Parameters
    ----------
    normalize: if True, divide by the area of the perfect Qini curve so the
               coefficient lies in [0, 1] for a perfect model and ≈0 for random.
               If False, return the raw area (useful for cross-dataset comparison).
    """
    fractions, gain, perfect_gain = _qini_curve(uplift_score, treatment, outcome)
    # Random baseline: straight line from (0,0) to (1, gain[-1])
    random_gain = fractions * gain[-1]
    model_area = np.trapz(gain, fractions)
    random_area = np.trapz(random_gain, fractions)
    qini = model_area - random_area

    if normalize:
        perfect_area = np.trapz(perfect_gain, fractions) - random_area
        if perfect_area <= 0:
            return 0.0
        return qini / perfect_area
    return qini


# ---------------------------------------------------------------------------
# AUUC — Area Under the Uplift Curve
# ---------------------------------------------------------------------------


def _uplift_curve(uplift_score, treatment, outcome):
    """
    Returns (fractions, uplift_values).

    The uplift curve plots E[Y(1) - Y(0) | score >= threshold] vs fraction
    targeted, estimated by comparing treated vs control mean outcomes in the
    top-k population.
    """
    uplift_score, treatment, outcome = _to_arrays(uplift_score, treatment, outcome)
    n = len(uplift_score)
    order = np.argsort(-uplift_score)
    t = treatment[order]
    y = outcome[order]

    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)

    with np.errstate(invalid="ignore", divide="ignore"):
        yt_mean = np.where(cum_t > 0, cum_yt / cum_t, np.nan)
        yc_mean = np.where(cum_c > 0, cum_yc / cum_c, np.nan)
    uplift_vals = yt_mean - yc_mean

    fractions = np.arange(1, n + 1) / n
    return fractions, uplift_vals


def auuc(uplift_score, treatment, outcome) -> float:
    """Area Under the Uplift Curve (vs random targeting baseline)."""
    fractions, uplift_vals = _uplift_curve(uplift_score, treatment, outcome)
    # Fill NaN at beginning with 0
    uplift_vals = np.where(np.isnan(uplift_vals), 0.0, uplift_vals)
    model_area = np.trapz(uplift_vals, fractions)
    # Random baseline AUUC: average treatment effect across full population
    ate = float(np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0]))
    random_area = ate * 0.5  # triangle under constant ATE line
    return float(model_area - random_area)


# ---------------------------------------------------------------------------
# Uplift @ k
# ---------------------------------------------------------------------------


def uplift_at_k(uplift_score, treatment, outcome, k: float = 0.3) -> float:
    """Estimated ATE in the top-k fraction of the population by predicted uplift.

    k: fraction of population to target (0 < k <= 1).
    """
    uplift_score, treatment, outcome = _to_arrays(uplift_score, treatment, outcome)
    n = len(uplift_score)
    cutoff = max(1, int(np.ceil(k * n)))
    order = np.argsort(-uplift_score)
    idx = order[:cutoff]
    t_k = treatment[idx]
    y_k = outcome[idx]
    n_t = t_k.sum()
    n_c = (1 - t_k).sum()
    if n_t == 0 or n_c == 0:
        return float("nan")
    return float(y_k[t_k == 1].mean() - y_k[t_k == 0].mean())


# ---------------------------------------------------------------------------
# Policy value at budget k%
# ---------------------------------------------------------------------------


def policy_value_at_k(
    uplift_score,
    treatment,
    outcome,
    k: float = 0.3,
    propensity: float | np.ndarray | None = None,
) -> float:
    """Estimated policy value when treating the top-k% of the population.

    Uses inverse-probability weighting (IPW) if propensity is provided,
    otherwise uses the simple difference-in-means estimator.

    policy_value = E[Y | T=1, score >= threshold] * P(score >= threshold)
                 + E[Y | T=0, score < threshold] * P(score < threshold)

    Parameters
    ----------
    k          : fraction targeted (treated), 0 < k <= 1
    propensity : scalar or array of P(T=1|X); enables IPW correction.
                 For RCTs with constant propensity, pass a scalar.
    """
    uplift_score, treatment, outcome = _to_arrays(uplift_score, treatment, outcome)
    n = len(uplift_score)
    cutoff = max(1, int(np.ceil(k * n)))
    order = np.argsort(-uplift_score)

    # Policy: treat top-k, don't treat bottom (1-k)
    policy = np.zeros(n, dtype=float)
    policy[order[:cutoff]] = 1.0

    if propensity is None:
        # Naïve: use empirical means within matched cells
        mask_treat = (policy == 1) & (treatment == 1)
        mask_ctrl = (policy == 0) & (treatment == 0)
        if mask_treat.sum() == 0 or mask_ctrl.sum() == 0:
            return float("nan")
        val = float(k * outcome[mask_treat].mean() + (1 - k) * outcome[mask_ctrl].mean())
    else:
        # IPW estimator
        prop = np.broadcast_to(np.asarray(propensity, dtype=float), (n,))
        # Value of treating top-k: E[Y(1)] estimated via IPW on treated
        # Value of not treating bottom (1-k): E[Y(0)] estimated via IPW on controls
        with np.errstate(invalid="ignore", divide="ignore"):
            y1_ipw = np.where(
                (policy == 1) & (treatment == 1),
                outcome / prop,
                0.0,
            )
            y0_ipw = np.where(
                (policy == 0) & (treatment == 0),
                outcome / (1 - prop),
                0.0,
            )
        val = float(y1_ipw.sum() / n + y0_ipw.sum() / n)

    return val
