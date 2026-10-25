# FILE: src/signals/impairments.py
# What this file does:
#   - Holds the channel/receiver impairments applied to clean IQ signals.
#   - add_awgn: adds complex white Gaussian noise at an exact target SNR,
#     where SNR = mean(|x|^2) / noise power (same function you tested in Cell 7).
#   - apply_freq_offset: multiplies the signal by exp(j*(2*pi*df*n/fs + phi)),
#     which shifts its spectrum by df Hz and rotates its phase by phi. This
#     simulates receiver oscillator error (carrier frequency/phase offset).
#   - Every function that uses randomness takes an explicit rng argument, so
#     experiments are reproducible.

import numpy as np


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator,
             ref_power: float | None = None) -> np.ndarray:
    """Return x plus complex white Gaussian noise at the given SNR (dB).
    SNR = ref_power / noise power. If ref_power is None, ref_power = mean(|x|^2)."""
    p_signal = np.mean(np.abs(x) ** 2) if ref_power is None else ref_power
    p_noise = p_signal / (10 ** (snr_db / 10))
    noise = np.sqrt(p_noise / 2) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )
    return x + noise


def apply_freq_offset(x: np.ndarray, offset_hz: float, fs: float,
                      phase_rad: float = 0.0) -> np.ndarray:
    """Shift the signal's spectrum by offset_hz and rotate its phase by phase_rad."""
    n = np.arange(len(x))
    return x * np.exp(1j * (2 * np.pi * offset_hz * n / fs + phase_rad))