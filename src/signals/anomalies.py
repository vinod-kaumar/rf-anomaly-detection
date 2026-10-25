# FILE: src/signals/anomalies.py
# What this file does:
#   - Generators for the anomaly families (all return complex I/Q arrays):
#       tone            narrowband / CW interference
#       wideband_noise  noise-like burst (jamming-style, flat spectrum)
#       chirp           linear frequency sweep
#       hopping         tone that jumps between a set of frequencies
#       apply_freq_drift  makes an existing signal's carrier drift
#   - inject: places an anomaly into a CLEAN normal signal at a chosen start
#     sample and returns (signal, label_mask). The label mask is True on every
#     sample the anomaly covers. Two modes:
#       "add"     the anomaly is added on top of the normal signal (interference)
#       "replace" the anomaly replaces the normal signal in that window
#                 (a drifted copy or a swapped modulation)
#   - Anomaly power is set relative to a unit-power normal signal (rel_power_db),
#     so it does not depend on the noise level. Add noise AFTER injecting, with
#     add_awgn(..., ref_power=1.0), so the SNR is defined by the normal signal only.

import numpy as np


def tone(n: int, fs: float, freq_hz: float, phase: float = 0.0) -> np.ndarray:
    """Complex tone at freq_hz, unit power."""
    return np.exp(1j * (2 * np.pi * freq_hz * np.arange(n) / fs + phase))


def wideband_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Complex white Gaussian noise, unit power."""
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def chirp(n: int, fs: float, f_start_hz: float, f_stop_hz: float) -> np.ndarray:
    """Linear sweep from f_start_hz to f_stop_hz over n samples, unit power."""
    t = np.arange(n) / fs
    sweep_rate = (f_stop_hz - f_start_hz) / (n / fs)       # Hz per second
    return np.exp(1j * 2 * np.pi * (f_start_hz * t + 0.5 * sweep_rate * t ** 2))


def hopping(n: int, fs: float, freqs_hz, dwell: int, rng: np.random.Generator) -> np.ndarray:
    """Tone that jumps to a randomly chosen frequency from freqs_hz every `dwell`
    samples (phase-continuous), unit power."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n_hops = int(np.ceil(n / dwell))
    f = np.repeat(freqs_hz[rng.integers(0, len(freqs_hz), n_hops)], dwell)[:n]
    return np.exp(1j * 2 * np.pi * np.cumsum(f) / fs)


def apply_freq_drift(x: np.ndarray, fs: float, total_drift_hz: float) -> np.ndarray:
    """Make x's carrier drift linearly from +0 Hz to +total_drift_hz across its length."""
    f_inst = np.linspace(0.0, total_drift_hz, len(x))
    return x * np.exp(1j * 2 * np.pi * np.cumsum(f_inst) / fs)


def inject(base: np.ndarray, anomaly: np.ndarray, start: int, rel_power_db: float = 0.0,
           mode: str = "add", ref_power: float = 1.0):
    """Place `anomaly` into `base` starting at sample `start`.
    The anomaly is rescaled so its mean power equals ref_power * 10^(rel_power_db/10).
    Returns (y, mask): y is the new signal, mask is a boolean array, True where the
    anomaly is present."""
    n, length = len(base), len(anomaly)
    if mode not in ("add", "replace"):
        raise ValueError("mode must be 'add' or 'replace'")
    if length == 0 or start < 0 or start + length > n:
        raise ValueError("anomaly must fit inside the base signal")
    p_anomaly = np.mean(np.abs(anomaly) ** 2)
    seg = anomaly * np.sqrt(ref_power * 10 ** (rel_power_db / 10) / p_anomaly)
    y = np.array(base, dtype=complex)
    if mode == "add":
        y[start:start + length] += seg
    else:
        y[start:start + length] = seg
    mask = np.zeros(n, dtype=bool)
    mask[start:start + length] = True
    return y, mask