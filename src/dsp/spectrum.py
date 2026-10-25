# FILE: src/dsp/spectrum.py
# What this file does:
#   - spectrogram_db: the project's one standard way to turn I/Q samples into a
#     spectrogram. Uses a Hann window, frequency axis centred on 0 Hz
#     (-fs/2 .. +fs/2), and power spectral density scaling.
#   - Returns (t, f, S_db): time axis in seconds, frequency axis in Hz and the
#     spectrogram in dB/Hz with shape (n_freq_bins, n_time_slices).
#   - Keeping this in one place means the notebooks, the models and the
#     dashboard all see exactly the same spectrogram.

import numpy as np
from scipy import signal


def spectrogram_db(iq: np.ndarray, fs: float, win_len: int = 256, hop: int = 64):
    sft = signal.ShortTimeFFT(signal.get_window("hann", win_len), hop=hop, fs=fs,
                              fft_mode="centered", scale_to="psd")
    s = sft.spectrogram(iq)
    return sft.t(len(iq)), sft.f, 10 * np.log10(s + 1e-20)