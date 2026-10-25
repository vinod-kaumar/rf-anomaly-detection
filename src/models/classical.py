# FILE: src/models/classical.py
# What this file does:
#   - Classical machine-learning anomaly detectors that work on the 13 window
#     features. They are trained ONLY on normal windows and learn the region of
#     feature space that normal windows occupy.
#   - FeatureDetector: a small wrapper that (1) optionally keeps only some
#     features, (2) standardises them (mean 0, std 1, learned from training
#     data), (3) fits the model. score() returns an anomaly score where HIGHER
#     = MORE ANOMALOUS, the convention used by src/eval/metrics.py.
#   - isolation_forest(): Isolation Forest. Anomalies are easier to isolate with
#     random splits, so they sit in shallow tree leaves.
#   - one_class_svm(): One-Class SVM with an RBF kernel. Learns a boundary
#     around the normal data; nu is roughly the fraction of training data
#     allowed outside it.
#   - Both accept feature_names=[...] to use a subset of features (for ablations).

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.dsp.features import FEATURE_NAMES


class FeatureDetector:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.columns = (list(range(len(FEATURE_NAMES))) if feature_names is None
                        else [FEATURE_NAMES.index(n) for n in feature_names])
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, feats: np.ndarray):
        x = self.scaler.fit_transform(np.asarray(feats)[:, self.columns])
        self.model.fit(x)
        self._fitted = True
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before score()")
        x = self.scaler.transform(np.asarray(feats)[:, self.columns])
        return -self.model.score_samples(x)


def isolation_forest(n_estimators: int = 300, seed: int = 0, feature_names=None, **kwargs):
    return FeatureDetector(IsolationForest(n_estimators=n_estimators, random_state=seed, **kwargs),
                           feature_names)


def one_class_svm(nu: float = 0.05, gamma="scale", feature_names=None, **kwargs):
    return FeatureDetector(OneClassSVM(nu=nu, gamma=gamma, **kwargs), feature_names)
