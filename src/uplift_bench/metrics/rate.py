"""RATE: rank-weighted average treatment effect (Yadlowsky et al., 2021/2025).

Evaluates a treatment-prioritization *ranking* against unbiased effect scores,
without requiring known individual effects. For unit i with treatment T_i in
{0,1}, outcome Y_i, and (known or estimated) propensity e_i, the IPW effect
score

    Gamma_i = T_i Y_i / e_i - (1 - T_i) Y_i / (1 - e_i)

satisfies E[Gamma_i | X_i] = tau(X_i). Ranking units by a model's uplift score
S_i (descending), the Targeting Operator Characteristic at treated fraction
q = j/n is

    TOC(q) = mean(Gamma over the top-j ranked units) - mean(Gamma over all),

and RATE is the weighted area under the TOC:

    RATE = (1/n) * sum_j w(j/n) * TOC(j/n),

with w = 1 giving AUTOC (uniform weighting; sensitive to targeting quality at
small fractions) and w(q) = q giving the Qini-weighted RATE of the grf
convention. Ties in S are handled by averaging Gamma within tie groups, so the
statistic is invariant to the order of tied units.

This is the benchmark-side implementation used to answer reviewer question
"do recently proposed metrics fix F1/F2?". We use the IPW score
(propensities are known for the RCTs and estimated on training folds
elsewhere, exactly as for the other metrics); the doubly robust variant would
additionally require outcome-model nuisances at metric-evaluation time, which
the benchmark deliberately avoids.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ipw_scores", "rate"]


def ipw_scores(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    clip: float = 0.01,
) -> np.ndarray:
    """Unbiased IPW effect scores Gamma_i; propensities clipped to [clip, 1-clip]."""
    t = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    e = np.clip(np.asarray(propensity, dtype=float), clip, 1.0 - clip)
    return t * y / e - (1.0 - t) * y / (1.0 - e)


def rate(
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    weighting: str = "autoc",
    clip: float = 0.01,
) -> float:
    """Rank-weighted ATE of the ranking induced by ``uplift_score``.

    Parameters
    ----------
    uplift_score: model prioritization scores (higher = treat first)
    treatment, outcome, propensity: evaluation-fold data
    weighting: "autoc" (uniform) or "qini" (w(q) = q)
    clip: propensity clipping for the IPW scores

    Returns the RATE estimate (0 for an uninformative ranking, in expectation).
    """
    if weighting not in ("autoc", "qini"):
        raise ValueError(f"weighting must be 'autoc' or 'qini', got {weighting!r}")

    s = np.asarray(uplift_score, dtype=float)
    gamma = ipw_scores(treatment, outcome, propensity, clip=clip)
    n = len(s)
    if n == 0 or len(gamma) != n:
        raise ValueError("uplift_score and treatment/outcome/propensity must share length")

    # Sort descending by score; average Gamma within tie groups so tied units
    # contribute their group mean at every rank they span.
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    g_sorted = gamma[order]
    # tie-group means
    _, group_idx, counts = np.unique(-s_sorted, return_inverse=True, return_counts=True)
    group_sums = np.bincount(group_idx, weights=g_sorted)
    g_sorted = (group_sums / counts)[group_idx]

    csum = np.cumsum(g_sorted)
    j = np.arange(1, n + 1)
    toc = csum / j - gamma.mean()
    w = np.ones(n) if weighting == "autoc" else j / n
    return float(np.sum(w * toc) / n)
