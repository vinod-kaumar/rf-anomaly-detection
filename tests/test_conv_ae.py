# FILE: tests/test_conv_ae.py
# What this file does:
#   - Checks ConvAEDetector: error_map shape [n, 32, 32]; score() shape; the
#     "max" score is never below the "mean" score; it separates structured
#     pictures from random noise (AUROC > 0.95); raises a clear error before
#     fit() and for a bad mode.
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.eval.metrics import auroc
from src.models.conv_ae import ConvAEDetector


def structured(rng, n):
    phase = rng.uniform(0, 2 * np.pi, n)[:, None]
    rows = np.cos(2 * np.pi * np.arange(32)[None, :] / 32 + phase)       # [n, 32]
    img = np.repeat(rows[:, :, None], 32, axis=2) + 0.05 * rng.standard_normal((n, 32, 32))
    return img.reshape(n, -1)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    return ConvAEDetector(bottleneck=8, epochs=40).fit(structured(rng, 600)), rng


def test_shapes_and_max_vs_mean(fitted):
    det, rng = fitted
    x = structured(rng, 20)
    assert det.error_map(x).shape == (20, 32, 32)
    mean, mx = det.score(x, "mean"), det.score(x, "max")
    assert mean.shape == (20,) and np.all(mx >= mean - 1e-9)


def test_separates_structured_from_noise(fitted):
    det, rng = fitted
    noise = rng.standard_normal((200, 1024))
    assert auroc(det.score(structured(rng, 200)), det.score(noise)) > 0.95


def test_errors_raise(fitted):
    det, rng = fitted
    with pytest.raises(RuntimeError):
        ConvAEDetector().score(np.zeros((2, 1024)))
    with pytest.raises(ValueError):
        det.score(structured(rng, 2), "median")