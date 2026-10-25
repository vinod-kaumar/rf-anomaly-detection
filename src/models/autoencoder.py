# FILE: src/models/autoencoder.py
# What this file does:
#   - AutoencoderDetector: a neural network that squeezes the input through a
#     narrow middle layer (16 numbers) and tries to rebuild it. Trained ONLY on
#     normal windows, so it rebuilds normal pictures well and unusual ones badly.
#   - Inputs are standardised with training statistics.
#   - score() = mean squared reconstruction error; HIGHER = MORE ANOMALOUS,
#     the same convention as src/eval/metrics.py.

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class AutoencoderDetector:
    def __init__(self, hidden=(256, 64, 16, 64, 256), seed: int = 0, max_iter: int = 300):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=hidden, early_stopping=True,
                                  validation_fraction=0.1, n_iter_no_change=15,
                                  max_iter=max_iter, random_state=seed)
        self._fitted = False

    def fit(self, x: np.ndarray):
        z = self.scaler.fit_transform(np.asarray(x))
        self.model.fit(z, z)
        self._fitted = True
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before score()")
        z = self.scaler.transform(np.asarray(x))
        return ((self.model.predict(z) - z) ** 2).mean(axis=1)