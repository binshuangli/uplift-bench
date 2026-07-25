"""Simple baseline uplift estimators via scikit-uplift.

Wrapped estimators
------------------
- ClassTransformationEstimator — class-variable transformation (Lai/Kane)
- TwoModelEstimator            — two-model (T-learner) via scikit-uplift
- SoloModelEstimator           — S-learner via scikit-uplift
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from uplift_bench.models.base import UpliftEstimator
from uplift_bench.models.base_learners import make_base_learner


class ClassTransformationEstimator(UpliftEstimator):
    """Class-variable transformation (Lai 2006 / Kane 2014).

    Requires binary outcomes; transforms the problem into a single
    classification task: Z = Y*T + (1-Y)*(1-T), then CATE ≈ 2*P(Z=1|X) - 1.

    Uses scikit-uplift's ClassTransformation wrapper.
    """

    name = "class_transformation"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        seed: int = 42,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.seed = seed
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from sklift.models import ClassTransformation

        clf = make_base_learner(self.base_learner, "classification", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = ClassTransformation(estimator=clf)
        self._model.fit(X_arr, y_arr, t_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.predict(X_arr)


class TwoModelEstimator(UpliftEstimator):
    """Two-model (T-learner) baseline via scikit-uplift.

    Fits separate classifiers for treated and control, returns P(Y=1|T=1,X) - P(Y=1|T=0,X).
    Equivalent to a T-learner with classification base learners.
    """

    name = "two_model"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        seed: int = 42,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.seed = seed
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from copy import deepcopy

        from sklift.models import TwoModels

        clf = make_base_learner(self.base_learner, "classification", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = TwoModels(
            estimator_trmnt=clf,
            estimator_ctrl=deepcopy(clf),
            method="vanilla",
        )
        self._model.fit(X_arr, y_arr, t_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.predict(X_arr)


class SoloModelEstimator(UpliftEstimator):
    """S-learner (Solo Model) baseline via scikit-uplift.

    Includes treatment as a feature in a single classifier.
    """

    name = "solo_model"

    def __init__(
        self,
        base_learner: Literal["lightgbm", "xgboost"] = "lightgbm",
        seed: int = 42,
        **hparams: Any,
    ):
        self.base_learner = base_learner
        self.seed = seed
        self.hparams = hparams
        self._model = None

    def fit(self, X, treatment, outcome, propensity=None):
        from sklift.models import SoloModel

        clf = make_base_learner(self.base_learner, "classification", self.seed, **self.hparams)
        X_arr, t_arr, y_arr, _ = self._to_numpy(X, treatment, outcome, propensity)
        self._model = SoloModel(estimator=clf)
        self._model.fit(X_arr, y_arr, t_arr)
        return self

    def predict_uplift(self, X):
        X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
        return self._model.predict(X_arr)
