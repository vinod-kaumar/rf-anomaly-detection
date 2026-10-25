# FILE: tests/test_metrics.py
# What this file does:
#   - Checks the evaluation functions in src/eval/metrics.py on scores where
#     the right answer is known, because every result in the project depends on them.
#   - threshold_at_far: flags the requested fraction of normal scores; rejects
#     invalid false-alarm rates.
#   - detection_rate / false_alarm_rate: simple fractions above the threshold.
#   - auroc / auprc: 1.0 for perfect separation, 0.0 AUROC when reversed, about
#     0.5 for identical distributions.
#   - family_report: a well-separated family gets AUROC ~1, an indistinguishable
#     family gets ~0.5, and detection rates follow the threshold.
#   - rate_by_bin: counts and rates per bin, NaN for an empty bin.
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.eval.metrics import (auprc, auroc, detection_rate, false_alarm_rate, family_report,
                              rate_by_bin, threshold_at_far)


def test_threshold_at_far_flags_requested_fraction():
    normal = np.arange(1000, dtype=float)
    thr = threshold_at_far(normal, 0.01)
    assert false_alarm_rate(normal, thr) == pytest.approx(0.01, abs=0.002)


@pytest.mark.parametrize("far", [0.0, 1.0, -0.1, 1.5])
def test_threshold_at_far_rejects_bad_far(far):
    with pytest.raises(ValueError):
        threshold_at_far([1.0, 2.0], far)


def test_detection_rate():
    assert detection_rate([1, 2, 3, 4], 2.5) == 0.5


def test_auroc_auprc_extremes():
    normal, anomaly = np.arange(100.0), np.arange(100.0, 200.0)
    assert auroc(normal, anomaly) == 1.0
    assert auprc(normal, anomaly) == 1.0
    assert auroc(anomaly, normal) == 0.0
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(5000), rng.standard_normal(5000)
    assert auroc(a, b) == pytest.approx(0.5, abs=0.03)


def test_family_report():
    rng = np.random.default_rng(1)
    scores = np.r_[rng.standard_normal(2000), rng.normal(6, 1, 300), rng.standard_normal(300)]
    y = np.r_[np.zeros(2000), np.ones(600)].astype(int)
    family = np.array(["normal"] * 2000 + ["easy"] * 300 + ["hidden"] * 300)
    thr = threshold_at_far(scores[y == 0], 0.05)
    rep = family_report(scores, y, family, thr)
    assert set(rep) == {"easy", "hidden"}
    assert rep["easy"]["auroc"] > 0.99 and rep["easy"]["det_rate"] > 0.99
    assert rep["hidden"]["auroc"] == pytest.approx(0.5, abs=0.06)
    assert rep["hidden"]["det_rate"] == pytest.approx(0.05, abs=0.04)
    assert rep["easy"]["n"] == 300


def test_rate_by_bin():
    scores = np.array([1.0, 3.0, 3.0, 1.0, 3.0])
    values = np.array([0.0, 1.0, 1.5, 2.5, 2.7])
    out = rate_by_bin(scores, values, thr=2.0, edges=[0, 1, 2, 3, 4])
    assert out[0] == (0, 1, 0.0, 1)
    assert out[1] == (1, 2, 1.0, 2)
    assert out[2][2] == pytest.approx(0.5) and out[2][3] == 2
    assert np.isnan(out[3][2]) and out[3][3] == 0