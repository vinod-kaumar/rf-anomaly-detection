# FILE: tests/test_anomalies.py
# What this file does:
#   - Automatic checks for src/signals/anomalies.py and the new ref_power
#     option of add_awgn.
#   - tone: spectral peak at the requested frequency, unit power.
#   - chirp: instantaneous frequency starts at f_start, ends at f_stop and rises
#     steadily.
#   - hopping: only the requested frequencies appear, more than one is used,
#     and it holds each for `dwell` samples.
#   - apply_freq_drift: the instantaneous frequency moves by total_drift_hz and
#     the power is unchanged.
#   - wideband_noise: unit power, I and Q variances equal.
#   - inject: mask marks exactly the anomaly window; "add" leaves the outside
#     untouched and adds the requested power; "replace" overwrites the window;
#     bad arguments raise ValueError.
#   - add_awgn with ref_power sets the noise from the reference, not from x.
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.signals.anomalies import tone, wideband_noise, chirp, hopping, apply_freq_drift, inject
from src.signals.impairments import add_awgn

FS = 1_000_000


def inst_freq(x):
    """Instantaneous frequency in Hz from phase differences."""
    return np.angle(x[1:] * np.conj(x[:-1])) * FS / (2 * np.pi)


def test_tone_frequency_and_power():
    n = 10_000
    x = tone(n, FS, 123_400)
    freqs = np.fft.fftfreq(n, d=1 / FS)
    assert freqs[np.argmax(np.abs(np.fft.fft(x)))] == pytest.approx(123_400, abs=100)
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0)


def test_chirp_sweeps_linearly():
    x = chirp(10_000, FS, -300_000, 300_000)
    f = inst_freq(x)
    assert f[0] == pytest.approx(-300_000, abs=200)
    assert f[-1] == pytest.approx(300_000, abs=200)
    assert np.all(np.diff(f) > -1)                # rising (tiny numeric tolerance)
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0)


def test_hopping_uses_only_given_frequencies_and_dwell():
    freqs = np.array([-200_000, -100_000, 100_000, 200_000])
    x = hopping(10_000, FS, freqs, dwell=500, rng=np.random.default_rng(0))
    f = inst_freq(x)
    assert np.isclose(f[:, None], freqs[None, :], atol=1e-3).any(axis=1).all()
    assert len(np.unique(np.round(f))) > 1
    assert np.allclose(f[1:499], f[1])            # constant inside the first dwell
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0)


def test_drift_moves_frequency_and_keeps_power():
    x = tone(10_000, FS, 100_000)
    y = apply_freq_drift(x, FS, 60_000)
    f = inst_freq(y)
    assert f[0] == pytest.approx(100_000, abs=200)
    assert f[-1] == pytest.approx(160_000, abs=200)
    assert np.mean(np.abs(y) ** 2) == pytest.approx(1.0)


def test_wideband_noise_power_and_circularity():
    x = wideband_noise(200_000, np.random.default_rng(1))
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=0.02)
    assert np.var(x.real) == pytest.approx(np.var(x.imag), rel=0.03)


def test_inject_add_mode():
    base = tone(10_000, FS, 100_000)
    anomaly = tone(2_000, FS, -200_000)
    y, mask = inject(base, anomaly, start=3_000, rel_power_db=-3.0, mode="add")
    assert mask.sum() == 2_000 and mask[3_000] and mask[4_999] and not mask[5_000]
    assert np.array_equal(y[~mask], base[~mask])                    # outside untouched
    added = (y - base)[mask]
    assert np.mean(np.abs(added) ** 2) == pytest.approx(10 ** (-0.3))


def test_inject_replace_mode():
    base = tone(10_000, FS, 100_000)
    anomaly = tone(2_000, FS, -200_000)
    y, mask = inject(base, anomaly, start=0, rel_power_db=0.0, mode="replace")
    assert np.allclose(y[mask], anomaly)                            # unit power, unchanged
    assert np.array_equal(y[~mask], base[~mask])


@pytest.mark.parametrize("start,length", [(-1, 10), (9_995, 10), (0, 0)])
def test_inject_rejects_bad_arguments(start, length):
    with pytest.raises(ValueError):
        inject(np.ones(10_000, dtype=complex), np.ones(length, dtype=complex), start)


def test_inject_rejects_bad_mode():
    with pytest.raises(ValueError):
        inject(np.ones(100, dtype=complex), np.ones(10, dtype=complex), 0, mode="multiply")


def test_add_awgn_ref_power_ignores_loud_signal():
    x = 10 * np.ones(200_000, dtype=complex)                        # power 100
    y = add_awgn(x, 10, np.random.default_rng(0), ref_power=1.0)    # noise power 0.1
    assert np.mean(np.abs(y - x) ** 2) == pytest.approx(0.1, rel=0.02)