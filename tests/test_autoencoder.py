# FILE: tests/test_autoencoder.py
# What this file does:
#   - Checks window_spectrogram: output shape (n, 1024), finite values, and that
#     multiplying the signal by 10 does not change it (level removed).
#   - Checks AutoencoderDetector: separates structured normal data from random
#     data (AUROC > 0.95) and raises a clear error if score() comes before fit().
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.dsp.spectro import window_spectrogram
from src.eval.metrics import auroc
from src.models.autoencoder import AutoencoderDetector


def test_spectrogram_shape_and_finite():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4096)) + 1j * rng.standard_normal((3, 4096))
    s = window_spectrogram(x)
    assert s.shape == (3, 1024) and np.all(np.isfinite(s))


def test_spectrogram_ignores_overall_level():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 4096)) + 1j * rng.standard_normal((2, 4096))
    assert np.allclose(window_spectrogram(x), window_spectrogram(10 * x), atol=1e-4)


def test_autoencoder_separates_structured_from_random():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((5, 64))
    make = lambda n: rng.standard_normal((n, 5)) @ w + 0.05 * rng.standard_normal((n, 64))
    det = AutoencoderDetector(hidden=(32, 8, 32), max_iter=500).fit(make(600))
    assert auroc(det.score(make(300)), det.score(rng.standard_normal((300, 64)) * 3)) > 0.95


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        AutoencoderDetector().score(np.zeros((2, 8)))