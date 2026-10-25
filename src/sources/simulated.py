# FILE: src/sources/simulated.py
# What this file does:
#   - Builds complete labelled recordings: a random NORMAL signal, optionally
#     one injected ANOMALY, then noise.
#   - Normal signal: random type (BPSK / QPSK / 16-QAM / FSK), samples per
#     symbol (8, 10 or 16), RRC roll-off (0.2-0.5), carrier offset (+-150 kHz)
#     and SNR (-5 to 25 dB, measured against the normal signal over the full
#     sampled band).
#   - Anomaly families (each with randomised parameters):
#       tone, noise_burst, chirp, hopping   added on top (interference)
#       drift                               the normal signal's carrier drifts
#       mod_swap                            a different LINEAR modulation with
#                                           identical spectrum replaces the
#                                           normal one ("hidden" family)
#   - The anomaly covers 20-40% of the recording, starting between 15% and
#     85% of it.
#   - Family groups: TUNE_FAMILIES may be used for validation/tuning;
#     HELDOUT_FAMILIES appear only in the test set.
#   - SimulatedSource: wraps a Recording as an IQSource so it can be streamed
#     with read(n), exactly like a future SDR would be.
#   - All randomness comes from the rng you pass in, so results are reproducible.

import numpy as np

from src.signals.anomalies import (apply_freq_drift, chirp, hopping, inject, tone,
                                   wideband_noise)
from src.signals.impairments import add_awgn, apply_freq_offset
from src.signals import modulations as mod
from src.sources.base import IQSource, Recording

FS = 1_000_000
N_DEFAULT = 32_768

NORMAL_TYPES = ("bpsk", "qpsk", "qam16", "fsk")
LINEAR_TYPES = ("bpsk", "qpsk", "qam16")

TUNE_FAMILIES = ("tone", "noise_burst")
HELDOUT_FAMILIES = ("chirp", "hopping", "drift", "mod_swap")
FAMILIES = TUNE_FAMILIES + HELDOUT_FAMILIES


def _make_normal(kind: str, n: int, sps: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "fsk":
        return mod.fsk(n, sps, rng)
    return {"bpsk": mod.bpsk, "qpsk": mod.qpsk, "qam16": mod.qam16}[kind](n, sps, rng, beta=beta)


def make_recording(rng: np.random.Generator, family: str | None = None, n: int = N_DEFAULT,
                   fs: float = FS, snr_range=(-5.0, 25.0)) -> Recording:
    """Create one labelled recording. family=None gives a normal-only recording."""
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")

    sps = int(rng.choice([8, 10, 16]))
    beta = float(rng.uniform(0.2, 0.5))
    carrier = float(rng.uniform(-150e3, 150e3))
    snr_db = float(rng.uniform(*snr_range))
    kind = str(rng.choice(LINEAR_TYPES if family == "mod_swap" else NORMAL_TYPES))

    base = apply_freq_offset(_make_normal(kind, n, sps, beta, rng), carrier, fs)
    meta = {"family": family or "normal", "normal_type": kind, "sps": sps, "beta": beta,
            "carrier_hz": carrier, "snr_db": snr_db}
    clean, mask = base, np.zeros(n, dtype=bool)

    if family is not None:
        length = int(rng.uniform(0.2, 0.4) * n)
        start = int(rng.uniform(0.15 * n, 0.85 * n - length))
        rel_db = float(rng.uniform(-12.0, -2.0))
        mode = "add"
        params = {}

        if family == "tone":
            f = float(rng.uniform(-450e3, 450e3))
            anomaly, params = tone(length, fs, f), {"freq_hz": f}
        elif family == "noise_burst":
            anomaly = wideband_noise(length, rng)
        elif family == "chirp":
            while True:
                f0, f1 = (float(v) for v in rng.uniform(-450e3, 450e3, 2))
                if abs(f1 - f0) >= 200e3:
                    break
            anomaly, params = chirp(length, fs, f0, f1), {"f_start_hz": f0, "f_stop_hz": f1}
        elif family == "hopping":
            freqs = rng.uniform(-450e3, 450e3, int(rng.integers(4, 7)))
            dwell = int(rng.choice([250, 500, 1000]))
            anomaly = hopping(length, fs, freqs, dwell, rng)
            params = {"n_freqs": len(freqs), "dwell": dwell}
        elif family == "drift":
            total = float(rng.choice([-1.0, 1.0]) * rng.uniform(20e3, 100e3))
            anomaly = apply_freq_drift(base[start:start + length], fs, total)
            mode, rel_db, params = "replace", 0.0, {"total_drift_hz": total}
        else:  # mod_swap
            other = str(rng.choice([t for t in LINEAR_TYPES if t != kind]))
            seg = _make_normal(other, length, sps, beta, rng)
            anomaly = apply_freq_offset(seg, carrier, fs)
            mode, rel_db, params = "replace", 0.0, {"swapped_to": other}

        clean, mask = inject(base, anomaly, start, rel_power_db=rel_db, mode=mode)
        meta.update({"start": start, "length": length, "rel_power_db": rel_db,
                     "mode": mode, "params": params})

    noisy = add_awgn(clean, snr_db, rng, ref_power=1.0)
    return Recording(iq=noisy, fs=fs, mask=mask, meta=meta)


class SimulatedSource(IQSource):
    """Streams a Recording's samples through the IQSource interface."""

    def __init__(self, recording: Recording):
        self.recording = recording
        self.fs = recording.fs
        self._pos = 0

    def read(self, n_samples: int) -> np.ndarray:
        chunk = self.recording.iq[self._pos:self._pos + n_samples]
        self._pos += len(chunk)
        return chunk