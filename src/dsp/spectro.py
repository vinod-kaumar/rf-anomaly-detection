# FILE: src/dsp/spectro.py
# What this file does:
#   - Turns each window of I/Q samples [n_windows, 4096] into a log-spectrogram
#     "picture", flattened to one row per window.
#   - Defaults: 32 time frames x 32 frequency bins = 1024 numbers (frames of 128
#     samples, groups of 4 FFT bins averaged, power in dB).
#   - normalise=True subtracts the window's own average dB level, so overall
#     signal level (SNR) does not dominate; only the SHAPE remains.
#   - Options exist for ablations: n_frames * frame must not exceed the window
#     length, and pool must divide frame.

import numpy as np


def window_spectrogram(x: np.ndarray, n_frames: int = 32, frame: int = 128,
                       pool: int = 4, normalise: bool = True) -> np.ndarray:
    x = np.atleast_2d(x).astype(np.complex64)
    n = x.shape[0]
    if n_frames * frame > x.shape[1] or frame % pool:
        raise ValueError("n_frames * frame must fit in the window and pool must divide frame")
    frames = x[:, : n_frames * frame].reshape(n, n_frames, frame) * np.hanning(frame).astype(np.float32)
    p = np.abs(np.fft.fftshift(np.fft.fft(frames, axis=-1), axes=-1)) ** 2
    p = p.reshape(n, n_frames, frame // pool, pool).mean(axis=-1)
    db = 10 * np.log10(p + 1e-12)
    if normalise:
        db -= db.mean(axis=(1, 2), keepdims=True)
    return db.reshape(n, -1).astype(np.float64)