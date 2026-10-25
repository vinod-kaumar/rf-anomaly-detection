# FILE: src/models/fusion.py
# What this file does:
#   - ScoreFusion: merges the scores of several detectors into one score.
#   - fit(): for each detector, learns the median and the (95th percentile minus
#     median) of its scores on NORMAL validation windows. No anomalies are used.
#   - combine(): rescales each detector's scores with those numbers, so a
#     typical normal window gets about 0 and a "5% unusual" normal window gets
#     about 1, then merges them with mode "max" (any detector alarmed) or
#     "mean" (average). HIGHER = MORE ANOMALOUS, as in src/eval/metrics.py.

import numpy as np


class ScoreFusion:
    def __init__(self, mode: str = "max"):
        if mode not in ("max", "mean"):
            raise ValueError("mode must be 'max' or 'mean'")
        self.mode = mode
        self.med, self.scale = [], []

    def fit(self, normal_scores):
        self.med, self.scale = [], []
        for s in normal_scores:
            s = np.asarray(s, dtype=float)
            m = np.median(s)
            self.med.append(m)
            self.scale.append(max(np.quantile(s, 0.95) - m, 1e-12))
        return self

    def combine(self, scores) -> np.ndarray:
        if len(scores) != len(self.med):
            raise ValueError("number of score arrays does not match fit()")
        z = np.stack([(np.asarray(s, dtype=float) - m) / sc
                      for s, m, sc in zip(scores, self.med, self.scale)])
        return z.max(axis=0) if self.mode == "max" else z.mean(axis=0)