# FILE: src/eval/seeds.py
# What this file does:
#   - mean_std(results): takes a list of dicts, one per random seed, each
#     {family: value} (e.g. AUROC), and returns {family: (mean, std)}.
#   - std is the sample standard deviation (n-1). With one seed it is 0.

import numpy as np


def mean_std(results: list) -> dict:
    out = {}
    for key in results[0]:
        v = np.array([r[key] for r in results], dtype=float)
        out[key] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return out