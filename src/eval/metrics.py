# FILE: src/eval/metrics.py
# What this file does:
#   - The evaluation toolkit used for EVERY detector, so comparisons are fair.
#   - Convention: a detector gives each window a score, HIGHER = MORE ANOMALOUS;
#     a window is flagged when score > threshold.
#   - threshold_at_far: threshold that flags a chosen fraction (the false-alarm
#     rate, FAR) of NORMAL windows. Always computed on validation normals,
#     never on test data.
#   - detection_rate / false_alarm_rate: fraction of anomalous / normal windows
#     flagged at a threshold.
#   - auroc / auprc: threshold-free measures of how well scores separate
#     normal from anomalous windows.
#   - family_report: AUROC, AUPRC and detection rate for each anomaly family
#     (each family is compared against ALL normal windows).
#   - rate_by_bin: a rate (detection or false-alarm) split into bins of a
#     variable, e.g. SNR.
#   - Windows labelled -1 (ambiguous) must be removed BEFORE calling these.

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_far(normal_scores, far: float) -> float:
    """Threshold above which a fraction `far` of the normal scores lies."""
    if not 0 < far < 1:
        raise ValueError("far must be between 0 and 1")
    return float(np.quantile(np.asarray(normal_scores), 1 - far))


def detection_rate(anomaly_scores, thr: float) -> float:
    return float(np.mean(np.asarray(anomaly_scores) > thr))


def false_alarm_rate(normal_scores, thr: float) -> float:
    return float(np.mean(np.asarray(normal_scores) > thr))


def _labels_scores(normal_scores, anomaly_scores):
    normal_scores, anomaly_scores = np.asarray(normal_scores), np.asarray(anomaly_scores)
    y = np.r_[np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))]
    return y, np.r_[normal_scores, anomaly_scores]


def auroc(normal_scores, anomaly_scores) -> float:
    y, s = _labels_scores(normal_scores, anomaly_scores)
    return float(roc_auc_score(y, s))


def auprc(normal_scores, anomaly_scores) -> float:
    y, s = _labels_scores(normal_scores, anomaly_scores)
    return float(average_precision_score(y, s))


def family_report(scores, y, family, thr: float) -> dict:
    """Per-family metrics. Returns {family: {"n", "auroc", "auprc", "det_rate"}}."""
    scores, y, family = np.asarray(scores), np.asarray(y), np.asarray(family)
    normal = scores[y == 0]
    report = {}
    for fam in sorted(set(family[y == 1])):
        anom = scores[(y == 1) & (family == fam)]
        report[fam] = {"n": len(anom), "auroc": auroc(normal, anom),
                       "auprc": auprc(normal, anom), "det_rate": detection_rate(anom, thr)}
    return report


def rate_by_bin(scores, values, thr: float, edges):
    """Fraction of scores above thr within each bin of `values`.
    Returns a list of (bin_low, bin_high, rate, n); rate is nan for empty bins."""
    scores, values = np.asarray(scores), np.asarray(values)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (values >= lo) & (values < hi)
        out.append((lo, hi, float(np.mean(scores[sel] > thr)) if sel.any() else float("nan"),
                    int(sel.sum())))
    return out