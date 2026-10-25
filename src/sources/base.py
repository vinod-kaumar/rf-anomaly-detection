# FILE: src/sources/base.py
# What this file does:
#   - Defines the ONE interface between "where the samples come from" and
#     everything downstream (DSP, models, dashboard).
#   - IQSource: anything that can hand out complex I/Q samples with
#     read(n_samples). A simulator, a SigMF file reader, or a real SDR can
#     all implement it, so the detector never needs to know the origin.
#   - Recording: a finished capture = the I/Q samples, the sample rate, a
#     per-sample boolean anomaly mask (ground truth, True where an anomaly
#     is present) and a dict of metadata describing how it was generated.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class IQSource(ABC):
    """Anything that can supply complex baseband samples."""

    fs: float                                   # sample rate in Hz

    @abstractmethod
    def read(self, n_samples: int) -> np.ndarray:
        """Return up to n_samples complex samples (fewer or none at the end)."""


@dataclass
class Recording:
    iq: np.ndarray                              # complex samples
    fs: float                                   # sample rate (Hz)
    mask: np.ndarray                            # bool, True where an anomaly is present
    meta: dict = field(default_factory=dict)    # how it was generated