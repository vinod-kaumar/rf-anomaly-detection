# FILE: tests/test_fusion.py
# What this file does:
#   - Checks ScoreFusion: after fit() the normal scores of each detector are
#     rescaled to median 0 and 95th percentile 1; "max" and "mean" merge as
#     expected; bad mode or wrong number of score arrays raises an error.
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.models.fusion import ScoreFusion


def test_rescaling_and_merging():
    rng = np.random.default_rng(0)
    a, b = rng.normal(5, 2, 5000), rng.normal(-3, 10, 5000)
    f = ScoreFusion("max").fit([a, b])
    za = (a - f.med[0]) / f.scale[0]
    assert np.median(za) == pytest.approx(0.0, abs=1e-9)
    assert np.quantile(za, 0.95) == pytest.approx(1.0, abs=1e-9)
    hi, lo = np.array([100.0]), np.array([-100.0])
    assert f.combine([hi, lo])[0] > 10                       # max follows the alarmed one
    mean = ScoreFusion("mean").fit([a, b])
    x = mean.combine([np.array([f.med[0]]), np.array([f.med[1]])])
    assert x[0] == pytest.approx(0.0, abs=1e-9)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        ScoreFusion("median")
    with pytest.raises(ValueError):
        ScoreFusion().fit([np.arange(10.0)]).combine([np.arange(3.0), np.arange(3.0)])