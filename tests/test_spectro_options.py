# FILE: tests/test_spectro_options.py
# What this file does:
#   - Checks the spectrogram options used in the ablation: output sizes for
#     frame 64 / 128 / 256, and that normalise=False keeps the signal level
#     (x10 amplitude = +20 dB) while normalise=True removes it.
#   Run with:  python -m pytest tests -v

import numpy as np

from src.dsp.spectro import window_spectrogram


def _x():
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 4096)) + 1j * rng.standard_normal((2, 4096))


def test_option_shapes():
    x = _x()
    assert window_spectrogram(x).shape == (2, 1024)
    assert window_spectrogram(x, n_frames=64, frame=64, pool=2).shape == (2, 2048)
    assert window_spectrogram(x, n_frames=16, frame=256, pool=8).shape == (2, 512)


def test_normalise_off_keeps_level():
    x = _x()
    diff = window_spectrogram(10 * x, normalise=False) - window_spectrogram(x, normalise=False)
    assert np.allclose(diff, 20.0, atol=1e-2)


def test_normalise_on_removes_level():
    x = _x()
    assert np.allclose(window_spectrogram(10 * x), window_spectrogram(x), atol=1e-3)