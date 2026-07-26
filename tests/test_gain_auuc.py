"""The cumulative-gain AUUC used in the weighting decomposition must match causalml.

`gain_auuc` (scripts/gain_auuc_check.py) integrates u(k)*k on the fraction axis;
`causalml.metrics.auuc_score(..., normalize=False)` integrates the cumulative gain
curve on the unit index. The identity is auuc_score == n * gain_auuc up to curve
discretization; this test pins both the numeric identity (within 2% on every fold)
and exact rank agreement across folds, so the manuscript's validation claim is
reproducible rather than anecdotal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from gain_auuc_check import gain_auuc  # noqa: E402

causalml_metrics = pytest.importorskip("causalml.metrics")


class TestGainAuucMatchesCausalml:
    def test_numeric_identity_and_rank_agreement(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(0)
        n = 200
        ours, theirs = [], []
        for _ in range(30):
            t = (rng.random(n) < 0.5).astype(float)
            score = rng.normal(size=n)
            # outcome with real uplift correlated with the score, so gain_auuc is
            # bounded away from 0 and the relative deviation is well-defined
            y = 0.5 * t * (1 + 0.5 * score) + rng.normal(scale=0.5, size=n)
            ga = gain_auuc(score, t, y)
            df = pd.DataFrame({"y": y, "w": t.astype(int), "score": score})
            cm = causalml_metrics.auuc_score(
                df, outcome_col="y", treatment_col="w", normalize=False
            )
            cm = float(cm["score"]) if hasattr(cm, "__getitem__") else float(cm)
            assert abs(cm / (n * ga) - 1.0) < 0.02, "auuc_score != n * gain_auuc"
            ours.append(ga)
            theirs.append(cm)
        assert spearmanr(ours, theirs).correlation > 0.9999
