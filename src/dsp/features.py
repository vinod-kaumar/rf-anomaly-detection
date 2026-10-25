# FILE: src/dsp/features.py
# What this file does:
#   - Turns each window of I/Q samples (shape [n_windows, window_len]) into a
#     vector of 13 hand-designed features, the input for the classical
#     detectors (energy / peak-to-median baselines, Isolation Forest, One-Class SVM).
#   - Spectral features come from a Welch PSD (Hann, 256-sample segments, 50%
#     overlap, NO detrending) estimated on each half of the window:
#       power_db            average power in dB (energy detector)
#       flatness            geometric/arithmetic mean of the PSD (1 = white noise)
#       centroid, spread    centre and width of the spectrum (cycles/sample)
#       peak_to_median_db   strongest PSD bin vs median bin (narrowband lines)
#       dynamic_range_db    90th vs 10th percentile PSD bin (how occupied the band is)
#       centroid_shift      spectral centroid of 2nd half minus 1st half (drift)
#       spectral_flux       mean |log PSD(1st half) - log PSD(2nd half)| (change over time)
#   - Time-domain features:
#       temporal_var        std/mean of the power in 16 sub-blocks (bursts)
#       env_kurtosis        E|x|^4 / (E|x|^2)^2 (1 = constant envelope, 2 = Gaussian noise)
#       papr_db             peak-to-average power ratio
#   - Modulation-structure features ("IQ-level", for signals with the same PSD):
#       line2_db, line4_db  strongest spectral line of x^2 and x^4 compared with the
#                           average of the bins just around it (+-4 to +-40 bins), so a
#                           smooth band does not count as a line. BPSK shows a clear
#                           line in x^2; QPSK, 16-QAM much weaker.
#   - Windows are processed in chunks to bound memory.

import numpy as np
from scipy import signal

FEATURE_NAMES = ["power_db", "flatness", "centroid", "spread", "peak_to_median_db",
                 "dynamic_range_db", "centroid_shift", "spectral_flux", "temporal_var",
                 "env_kurtosis", "papr_db", "line2_db", "line4_db"]

_SEG = 256
_EPS = 1e-20


def _psd(x: np.ndarray) -> np.ndarray:
    _, p = signal.welch(x, nperseg=_SEG, noverlap=_SEG // 2, detrend=False,
                        return_onesided=False, axis=-1)
    return np.fft.fftshift(p, axes=-1) + _EPS


def _line_db(z: np.ndarray) -> np.ndarray:
    """Strongest spectral line of z versus the mean of the bins around it (dB)."""
    s = np.abs(np.fft.fft(z, axis=1)) ** 2 + _EPS
    n, w = s.shape
    k = s.argmax(axis=1)
    offsets = np.r_[-40:-3, 4:41]
    around = np.take_along_axis(s, (k[:, None] + offsets[None, :]) % w, axis=1)
    return 10 * np.log10(s[np.arange(n), k] / around.mean(axis=1))


def _features_chunk(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.complex128)
    n, w = x.shape
    p1, p2 = _psd(x[:, : w // 2]), _psd(x[:, w // 2:])
    p = 0.5 * (p1 + p2)
    f = np.fft.fftshift(np.fft.fftfreq(p.shape[1]))
    ptot = p.sum(axis=1)

    centroid = (p * f).sum(axis=1) / ptot
    spread = np.sqrt((p * (f - centroid[:, None]) ** 2).sum(axis=1) / ptot)
    flatness = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
    peak_to_median = 10 * np.log10(p.max(axis=1) / np.median(p, axis=1))
    dyn = 10 * np.log10(np.percentile(p, 90, axis=1) / np.percentile(p, 10, axis=1))
    c1 = (p1 * f).sum(axis=1) / p1.sum(axis=1)
    c2 = (p2 * f).sum(axis=1) / p2.sum(axis=1)
    flux = np.mean(np.abs(np.log(p1) - np.log(p2)), axis=1)

    pw = np.abs(x) ** 2
    power = pw.mean(axis=1)
    blocks = pw.reshape(n, 16, -1).mean(axis=2)
    temporal_var = blocks.std(axis=1) / blocks.mean(axis=1)
    env_kurt = (pw ** 2).mean(axis=1) / power ** 2
    papr = 10 * np.log10(pw.max(axis=1) / power)

    return np.column_stack([10 * np.log10(power), flatness, centroid, spread, peak_to_median,
                            dyn, c2 - c1, flux, temporal_var, env_kurt, papr,
                            _line_db(x ** 2), _line_db(x ** 4)])


def window_features(x: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Features for windows x of shape [n_windows, window_len] -> [n_windows, 13]."""
    x = np.atleast_2d(x)
    return np.concatenate([_features_chunk(x[i:i + chunk]) for i in range(0, len(x), chunk)])