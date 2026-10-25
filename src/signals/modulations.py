# FILE: src/signals/modulations.py
# What this file does:
#   - Generates the "normal" signals for the project as complex baseband (I/Q)
#     samples with UNIT average power: BPSK, QPSK, 16-QAM and FSK.
#   - rrc_taps: root-raised-cosine (RRC) pulse-shaping filter with unit energy.
#     Shaping limits the bandwidth to about Rs*(1+beta) (Rs = symbol rate).
#   - pulse_shape: puts symbols on a sample grid (sps samples per symbol),
#     filters them with the RRC taps, and removes the filter delay so symbol
#     k sits at sample k*sps.
#   - bpsk / qpsk / qam16: draw random symbols, pulse-shape them, trim the
#     filter start/end transients, return exactly n_samples samples.
#   - fsk: continuous-phase FSK (constant envelope). Tone spacing is set by
#     the modulation index h (h = 1 -> tones at +-Rs/2 for binary FSK).
#   - Every function takes an explicit rng so results are reproducible.
#   - span (filter length in symbols) must be even.

import numpy as np


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine filter taps (length span*sps + 1), unit energy."""
    n = span * sps
    if n % 2:
        raise ValueError("span * sps must be even")
    t = np.arange(-(n // 2), n // 2 + 1) / sps          # time in symbol periods
    h = np.zeros_like(t)
    eps = 1e-8
    zero = np.abs(t) < eps                               # t = 0 special case
    if beta > 0:
        sing = np.abs(np.abs(t) - 1 / (4 * beta)) < eps  # t = +-1/(4 beta) special case
    else:
        sing = np.zeros_like(zero)
    reg = ~(zero | sing)

    h[zero] = 1 - beta + 4 * beta / np.pi
    if beta > 0:
        h[sing] = (beta / np.sqrt(2)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
        )
    tr = t[reg]
    h[reg] = (np.sin(np.pi * tr * (1 - beta)) + 4 * beta * tr * np.cos(np.pi * tr * (1 + beta))) / (
        np.pi * tr * (1 - (4 * beta * tr) ** 2)
    )
    return h / np.sqrt(np.sum(h ** 2))


def pulse_shape(symbols: np.ndarray, sps: int, beta: float = 0.35, span: int = 10) -> np.ndarray:
    """Upsample symbols by sps, filter with RRC, remove delay. Output length = len(symbols)*sps.
    Scaled so unit-power symbols give ~unit-power samples. The first/last span*sps/2
    samples are filter transients (partial overlap)."""
    h = rrc_taps(beta, sps, span)
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    y = np.convolve(up, h)
    delay = (len(h) - 1) // 2
    return y[delay:delay + len(up)] * np.sqrt(sps)


def _generate_linear(draw_symbols, n_samples: int, sps: int, beta: float, span: int) -> np.ndarray:
    """Shared recipe: draw symbols, shape them, trim transients, return n_samples samples."""
    if span % 2:
        raise ValueError("span must be even")
    n_sym = int(np.ceil(n_samples / sps)) + span
    y = pulse_shape(draw_symbols(n_sym), sps, beta, span)
    start = (span * sps) // 2                            # skip the start transient
    return y[start:start + n_samples]


def bpsk(n_samples: int, sps: int, rng: np.random.Generator, beta: float = 0.35, span: int = 10):
    """BPSK: symbols +-1."""
    draw = lambda n: (rng.integers(0, 2, n) * 2 - 1).astype(complex)
    return _generate_linear(draw, n_samples, sps, beta, span)


def qpsk(n_samples: int, sps: int, rng: np.random.Generator, beta: float = 0.35, span: int = 10):
    """QPSK: symbols (+-1 +- 1j)/sqrt(2)."""
    draw = lambda n: ((rng.integers(0, 2, n) * 2 - 1) + 1j * (rng.integers(0, 2, n) * 2 - 1)) / np.sqrt(2)
    return _generate_linear(draw, n_samples, sps, beta, span)


def qam16(n_samples: int, sps: int, rng: np.random.Generator, beta: float = 0.35, span: int = 10):
    """16-QAM: I and Q from {-3,-1,1,3}, scaled by 1/sqrt(10) for unit average power."""
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    draw = lambda n: (levels[rng.integers(0, 4, n)] + 1j * levels[rng.integers(0, 4, n)]) / np.sqrt(10)
    return _generate_linear(draw, n_samples, sps, beta, span)


def fsk(n_samples: int, sps: int, rng: np.random.Generator, m: int = 2, h: float = 1.0):
    """Continuous-phase M-ary FSK, constant envelope (|x| = 1).
    Symbol k in {0..m-1} sets the frequency (2k-(m-1)) * h * Rs / 2, Rs = fs/sps."""
    n_sym = int(np.ceil(n_samples / sps))
    k = rng.integers(0, m, n_sym)
    f_cycles_per_sample = (2 * k - (m - 1)) * h / (2 * sps)
    f = np.repeat(f_cycles_per_sample, sps)[:n_samples]
    return np.exp(1j * 2 * np.pi * np.cumsum(f))