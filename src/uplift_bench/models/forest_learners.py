"""Tree / forest direct uplift estimators.

Wrapped estimators
------------------
- CausalForestEstimator    — CausalForestDML (EconML); honest causal forest
- UpliftRFEstimator        — UpliftRandomForestClassifier (CausalML)
  with three splitting criteria: KL, Euclidean (ED), Chi2
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from uplift_bench.models.base import UpliftEstimator
from uplift_bench.models.base_learners import make_base_learner


class CausalForestEstimator(UpliftEstimator):
    """Honest Causal Forest via EconML's CausalForestDML.

    Uses LightGBM/XGBoost to partial out Y and T before fitting the forest.
    n_estimators must be divisible by subforest_size (default 4).
    """

    name = "causal_forest"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        n_estimators: int = 100,
        seed: int = 42,
        cv: int = 3,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.seed = seed
        self.cv = cv
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from econml.dml import CausalForestDML

        reg = make_base_learner(self.base_learner, "regression", self.seed, **self.hparams)
        # discrete_treatment=True requires a classifier nuisance for T (see meta_learners
        # R-Learner note; same external-audit fix; released parquets <= v1.3.0 used the
        # regressor nuisance).
        clf = make_base_learner(self.base_learner, "classification", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        # Ensure n_estimators is divisible by subforest_size=4
        n_est = max(100, (self.n_estimators // 4) * 4)
        self._model = CausalForestDML(
            model_y=reg,
            model_t=clf,
            n_estimators=n_est,
            cv=self.cv,
            discrete_treatment=True,
            random_state=self.seed,
        )
        self._model.fit(y_arr, t_arr, X=X_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.effect(X_arr)


class UpliftRFEstimator(UpliftEstimator):
    """CausalML UpliftRandomForestClassifier with configurable splitting criterion.

    Note: CausalML's UpliftRF expects treatment as a string array with a
    designated control label ('control'). We convert binary treatment internally.
    """

    name = "uplift_rf"

    def __init__(
        self,
        evaluationFunction: Literal["KL", "ED", "Chi"] = "KL",
        n_estimators: int = 100,
        max_depth: int = 8,
        min_samples_leaf: int = 50,
        min_samples_treatment: int = 10,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.evaluationFunction = evaluationFunction
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.seed = seed
        self.n_jobs = n_jobs
        self.name = f"uplift_rf_{evaluationFunction.lower()}"
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from causalml.inference.tree import UpliftRandomForestClassifier

        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        # CausalML expects string treatment labels with n_jobs=1 for stability
        t_str = np.where(t_arr == 1, "treatment", "control")
        self._model = UpliftRandomForestClassifier(
            evaluationFunction=self.evaluationFunction,
            control_name="control",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_treatment=self.min_samples_treatment,
            random_state=self.seed,
            n_jobs=1,  # parallel bootstrap in CausalML is fragile with pandas DataFrames
        )
        self._model.fit(X_arr, t_str, y_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        # Returns shape (n, 1) for a single treatment arm
        pred = self._model.predict(X_arr)
        return pred.ravel()
