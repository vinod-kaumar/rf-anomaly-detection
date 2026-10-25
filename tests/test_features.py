# FILE: tests/test_features.py
# What this file does:
#   - Checks that each feature in src/dsp/features.py responds to the right
#     physical property, using signals whose answer we know.
#   - Shape and finiteness; processing in chunks gives identical results.
#   - White noise: flat spectrum, Gaussian envelope (kurtosis ~2), steady power,
#     ~0 dB power for unit-power noise, no strong spectral peak.
#   - Tone: constant envelope (kurtosis ~1), very strong peak-to-median ratio,
#     low flatness.
#   - Gated (on/off) signal: large temporal variation.
#   - Drifting tone: positive centroid shift and large spectral flux; a static
#     tone has neither.
#   - Modulation structure (averaged over 10 windows): BPSK has a much stronger
#     x^2 line than QPSK; QPSK has a stronger x^4 line than 16-QAM.
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.dsp.features import FEATURE_NAMES, window_features
from src.signals import modulations as mod
from src.signals.anomalies import apply_freq_drift, tone
from src.signals.impairments import add_awgn, apply_freq_offset

FS, W = 1_000_000, 4096
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def noise_window(rng):
    return (rng.standard_normal(W) + 1j * rng.standard_normal(W)) / np.sqrt(2)


def test_shape_and_finite():
    rng = np.random.default_rng(0)
    x = np.stack([noise_window(rng) for _ in range(5)])
    f = window_features(x)
    assert f.shape == (5, len(FEATURE_NAMES)) == (5, 13)
    assert np.all(np.isfinite(f))


def test_chunking_does_not_change_results():
    rng = np.random.default_rng(1)
    x = np.stack([add_awgn(tone(W, FS, 80e3), 5, rng) for _ in range(10)])
    assert np.allclose(window_features(x, chunk=3), window_features(x, chunk=100))


def test_white_noise_features():
    rng = np.random.default_rng(2)
    f = window_features(np.stack([noise_window(rng) for _ in range(20)])).mean(axis=0)
    assert abs(f[IDX["power_db"]]) < 0.3
    assert f[IDX["flatness"]] > 0.9
    assert 1.8 < f[IDX["env_kurtosis"]] < 2.2
    assert f[IDX["temporal_var"]] < 0.15
    assert f[IDX["peak_to_median_db"]] < 6


def test_tone_features():
    rng = np.random.default_rng(3)
    x = np.stack([add_awgn(tone(W, FS, 100e3), 10, rng) for _ in range(10)])
    f = window_features(x).mean(axis=0)
    assert f[IDX["env_kurtosis"]] < 1.4
    assert f[IDX["peak_to_median_db"]] > 25
    assert f[IDX["flatness"]] < 0.3
    assert f[IDX["centroid"]] == pytest.approx(0.1, abs=0.02)      # 100 kHz / 1 MHz


def test_gated_signal_has_high_temporal_variation():
    rng = np.random.default_rng(4)
    x = apply_freq_offset(mod.qpsk(W, 8, rng), 60e3, FS)
    gated = x.copy()
    gated[1000:3000] = 0
    f_on = window_features(add_awgn(x, 20, rng, ref_power=1.0)[None, :])[0]
    f_gated = window_features(add_awgn(gated, 20, rng, ref_power=1.0)[None, :])[0]
    assert f_gated[IDX["temporal_var"]] > 0.5 > f_on[IDX["temporal_var"]]


def test_drift_changes_centroid_and_flux():
    rng = np.random.default_rng(5)
    static = np.stack([add_awgn(tone(W, FS, 100e3), 10, rng) for _ in range(10)])
    drifting = np.stack([add_awgn(apply_freq_drift(tone(W, FS, 100e3), FS, 100e3), 10, rng)
                         for _ in range(10)])
    fs_, fd = window_features(static).mean(axis=0), window_features(drifting).mean(axis=0)
    assert abs(fs_[IDX["centroid_shift"]]) < 0.005
    assert fd[IDX["centroid_shift"]] > 0.03
    assert fd[IDX["spectral_flux"]] > fs_[IDX["spectral_flux"]] + 0.3


def test_modulation_lines():
    rng = np.random.default_rng(6)

    def mean_features(gen):
        x = np.stack([add_awgn(apply_freq_offset(gen(W, 8, rng), 60e3, FS), 20, rng, ref_power=1.0)
                      for _ in range(10)])
        return window_features(x).mean(axis=0)

    fb, fq, fm = mean_features(mod.bpsk), mean_features(mod.qpsk), mean_features(mod.qam16)
    assert fb[IDX["line2_db"]] > fq[IDX["line2_db"]] + 10
    assert fb[IDX["line2_db"]] > fm[IDX["line2_db"]] + 10
    assert fq[IDX["line4_db"]] > fm[IDX["line4_db"]] + 5