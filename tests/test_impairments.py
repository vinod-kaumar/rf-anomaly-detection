# FILE: tests/test_impairments.py
# What this file does:
#   - Automatic checks for the functions in src/signals/impairments.py.
#   - test_awgn_hits_target_snr: measured SNR must be within 0.1 dB of the
#     target for several SNRs (200,000 samples makes the estimate tight).
#   - test_awgn_noise_is_circular: I and Q noise must have equal variance and
#     be uncorrelated (a correct complex Gaussian).
#   - test_awgn_reproducible: same seed gives identical output.
#   - test_freq_offset_shifts_peak: a 100 kHz tone offset by 50 kHz must
#     appear at 150 kHz.
#   - test_freq_offset_preserves_power: shifting must not change power.
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.signals.impairments import add_awgn, apply_freq_offset

FS = 1_000_000


def make_tone(n, f0=100_000):
    return np.exp(1j * 2 * np.pi * f0 * np.arange(n) / FS)


@pytest.mark.parametrize("snr_db", [-10, 0, 10, 20])
def test_awgn_hits_target_snr(snr_db):
    x = make_tone(200_000)
    y = add_awgn(x, snr_db, np.random.default_rng(0))
    noise = y - x
    measured = 10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(noise) ** 2))
    assert abs(measured - snr_db) < 0.1


def test_awgn_noise_is_circular():
    x = make_tone(200_000)
    noise = add_awgn(x, 0, np.random.default_rng(1)) - x
    var_i, var_q = np.var(noise.real), np.var(noise.imag)
    assert abs(var_i - var_q) / var_i < 0.05
    corr = np.corrcoef(noise.real, noise.imag)[0, 1]
    assert abs(corr) < 0.02


def test_awgn_reproducible():
    x = make_tone(1000)
    a = add_awgn(x, 5, np.random.default_rng(7))
    b = add_awgn(x, 5, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_freq_offset_shifts_peak():
    n = 10_000                                   # 100 Hz bins, tone on a bin
    y = apply_freq_offset(make_tone(n), 50_000, FS)
    freqs = np.fft.fftfreq(n, d=1 / FS)
    peak = freqs[np.argmax(np.abs(np.fft.fft(y)))]
    assert peak == pytest.approx(150_000, abs=1)


def test_freq_offset_preserves_power():
    x = make_tone(10_000)
    y = apply_freq_offset(x, 12_345, FS, phase_rad=0.7)
    assert np.mean(np.abs(y) ** 2) == pytest.approx(np.mean(np.abs(x) ** 2))