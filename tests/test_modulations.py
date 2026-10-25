# FILE: tests/test_modulations.py
# What this file does:
#   - Automatic checks for src/signals/modulations.py.
#   - RRC filter: unit energy, symmetric, and the RRC*RRC pair is a Nyquist
#     (zero-ISI) pulse: zero at every symbol instant except the centre.
#   - Matched-filter round trip: shaping then matched filtering then sampling
#     at symbol instants recovers the transmitted symbols.
#   - All generators: exact length, ~unit average power, reproducible with the
#     same seed.
#   - Spectrum: shaped BPSK keeps ~all its power inside +-Rs(1+beta)/2 (with a
#     small margin), the property that makes a signal "narrowband".
#   - FSK: constant envelope and instantaneous frequency takes only the
#     expected values (+-Rs/2 for binary FSK with h = 1).
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest
from scipy import signal

from src.signals.modulations import rrc_taps, pulse_shape, bpsk, qpsk, qam16, fsk

SPS = 8
GENERATORS = {"bpsk": bpsk, "qpsk": qpsk, "qam16": qam16, "fsk": fsk}


@pytest.mark.parametrize("beta", [0.2, 0.25, 0.35, 0.5])
def test_rrc_unit_energy_and_symmetric(beta):
    h = rrc_taps(beta, SPS, 10)
    assert np.sum(h ** 2) == pytest.approx(1.0)
    assert np.allclose(h, h[::-1])


@pytest.mark.parametrize("beta", [0.25, 0.35, 0.5])
def test_rrc_pair_is_nyquist(beta):
    h = rrc_taps(beta, SPS, 10)
    rc = np.convolve(h, h)
    centre = len(h) - 1
    assert rc[centre] == pytest.approx(1.0, abs=1e-6)
    for k in range(1, 5):
        assert abs(rc[centre + k * SPS]) < 0.02
        assert abs(rc[centre - k * SPS]) < 0.02


def test_matched_filter_recovers_symbols():
    rng = np.random.default_rng(3)
    symbols = (rng.integers(0, 2, 300) * 2 - 1 + 1j * (rng.integers(0, 2, 300) * 2 - 1)) / np.sqrt(2)
    y = pulse_shape(symbols, SPS, 0.35, 10)
    h = rrc_taps(0.35, SPS, 10)
    z = np.convolve(y, h)[(len(h) - 1) // 2:(len(h) - 1) // 2 + len(y)] / np.sqrt(SPS)
    recovered = z[::SPS]
    mid = slice(15, -15)                        # skip edge symbols (partial filter overlap)
    assert np.max(np.abs(recovered[mid] - symbols[mid])) < 0.05


@pytest.mark.parametrize("name", list(GENERATORS))
def test_length_and_unit_power(name):
    x = GENERATORS[name](100_001, SPS, np.random.default_rng(0))
    assert len(x) == 100_001
    assert np.iscomplexobj(x)
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize("name", list(GENERATORS))
def test_reproducible(name):
    a = GENERATORS[name](5000, SPS, np.random.default_rng(11))
    b = GENERATORS[name](5000, SPS, np.random.default_rng(11))
    c = GENERATORS[name](5000, SPS, np.random.default_rng(12))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bpsk_power_stays_in_occupied_band():
    beta = 0.35
    x = bpsk(200_000, SPS, np.random.default_rng(5), beta=beta)
    f, p = signal.welch(x, fs=1.0, nperseg=2048, return_onesided=False)
    half_bw = (1 + beta) / (2 * SPS)            # ideal edge in cycles/sample
    inside = np.abs(f) <= 1.1 * half_bw
    assert p[inside].sum() / p.sum() > 0.99


def test_fsk_constant_envelope_and_frequencies():
    x = fsk(20_000, SPS, np.random.default_rng(2), m=2, h=1.0)
    assert np.allclose(np.abs(x), 1.0)
    inst = np.angle(x[1:] * np.conj(x[:-1])) / (2 * np.pi)     # cycles/sample
    expected = 1.0 / (2 * SPS)                                  # +-Rs/2 / fs
    assert np.allclose(np.abs(inst), expected, rtol=1e-6)
    assert set(np.sign(inst)) == {-1.0, 1.0}