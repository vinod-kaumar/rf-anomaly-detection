# FILE: src/dsp/windowing.py
# What this file does:
#   - Cuts a long recording into fixed-length windows (the unit the detector
#     scores) and labels each window from the per-sample anomaly mask.
#   - Label rule, based on the fraction of the window covered by the anomaly:
#         0  = normal     (no overlap at all)
#         1  = anomalous  (overlap >= min_overlap_frac, default 50%)
#        -1  = ambiguous  (tiny partial overlap; dropped from scoring so it
#                          counts neither as a detection nor a false alarm)
#   - window_signal: windows of one recording.
#   - collect_windows: windows of many recordings, stacked into arrays together
#     with the family, SNR and recording id of each window.

import numpy as np


def window_signal(iq: np.ndarray, mask: np.ndarray, win: int, hop: int,
                  min_overlap_frac: float = 0.5):
    """Return (windows [n_win, win], labels [n_win], starts [n_win])."""
    starts = np.arange(0, len(iq) - win + 1, hop)
    idx = starts[:, None] + np.arange(win)[None, :]
    overlap = mask[idx].mean(axis=1)
    labels = np.full(len(starts), -1, dtype=int)
    labels[overlap == 0] = 0
    labels[overlap >= min_overlap_frac] = 1
    return iq[idx], labels, starts


def collect_windows(recordings, win: int, hop: int, min_overlap_frac: float = 0.5):
    """Stack windows of many recordings. Returns a dict of arrays:
    x (complex64), y, family, snr_db, rec_id."""
    xs, ys, fams, snrs, ids = [], [], [], [], []
    for i, rec in enumerate(recordings):
        w, y, _ = window_signal(rec.iq, rec.mask, win, hop, min_overlap_frac)
        xs.append(w.astype(np.complex64))
        ys.append(y)
        fams += [rec.meta["family"]] * len(y)
        snrs += [rec.meta["snr_db"]] * len(y)
        ids += [i] * len(y)
    return {"x": np.concatenate(xs), "y": np.concatenate(ys), "family": np.array(fams),
            "snr_db": np.array(snrs), "rec_id": np.array(ids)}