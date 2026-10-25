# FILE: tests/test_dataset.py
# What this file does:
#   - Checks the dataset-generation pipeline end to end.
#   - make_recording: normal recordings have an empty mask; every family
#     produces one contiguous anomaly covering 20-40% of the recording; same
#     seed -> identical recording, different seed -> different; unknown family
#     raises; the noise level matches the requested SNR.
#   - mod_swap is "hidden": the average power inside the anomaly equals the
#     power outside (within 1 dB) at high SNR, and the swapped type differs.
#   - SimulatedSource: streaming read() returns exactly the recording's samples.
#   - window_signal: label rule (0 normal / 1 anomalous / -1 ambiguous) on a
#     hand-made mask.
#   - splits: train is normal-only; val contains only tune families; test
#     contains all families; the family groups never overlap; splits use
#     different data.
#   - spectrogram_db: a tone shows up in the right frequency bin.
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.data.splits import build_split, split_families
from src.dsp.spectrum import spectrogram_db
from src.dsp.windowing import collect_windows, window_signal
from src.signals.anomalies import tone
from src.sources.simulated import (FAMILIES, HELDOUT_FAMILIES, N_DEFAULT, TUNE_FAMILIES,
                                   SimulatedSource, make_recording)


def test_normal_recording_has_empty_mask():
    rec = make_recording(np.random.default_rng(0), None)
    assert len(rec.iq) == N_DEFAULT and np.iscomplexobj(rec.iq)
    assert not rec.mask.any()
    assert rec.meta["family"] == "normal"


@pytest.mark.parametrize("family", FAMILIES)
def test_anomaly_mask_is_one_contiguous_block(family):
    for seed in range(5):
        rec = make_recording(np.random.default_rng(seed), family)
        m = rec.mask
        frac = m.mean()
        assert 0.19 <= frac <= 0.41
        edges = np.flatnonzero(np.diff(m.astype(int)))
        assert len(edges) == 2                              # exactly one on and one off edge
        assert rec.meta["family"] == family
        assert np.all(np.isfinite(rec.iq))


def test_reproducible_and_different_seeds():
    a = make_recording(np.random.default_rng(5), "chirp")
    b = make_recording(np.random.default_rng(5), "chirp")
    c = make_recording(np.random.default_rng(6), "chirp")
    assert np.array_equal(a.iq, b.iq) and np.array_equal(a.mask, b.mask)
    assert not np.array_equal(a.iq, c.iq)


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        make_recording(np.random.default_rng(0), "banana")


def test_noise_level_matches_snr():
    rng = np.random.default_rng(1)
    for _ in range(5):
        rec = make_recording(rng, None, n=65_536)
        expected = 1.0 + 10 ** (-rec.meta["snr_db"] / 10)
        assert np.mean(np.abs(rec.iq) ** 2) == pytest.approx(expected, rel=0.06)


def test_mod_swap_is_hidden_from_power_and_type_changes():
    for seed in range(6):
        rng = np.random.default_rng(seed)
        rec = make_recording(rng, "mod_swap", snr_range=(25.0, 25.0))
        p_in = np.mean(np.abs(rec.iq[rec.mask]) ** 2)
        p_out = np.mean(np.abs(rec.iq[~rec.mask]) ** 2)
        assert abs(10 * np.log10(p_in / p_out)) < 1.0
        assert rec.meta["params"]["swapped_to"] != rec.meta["normal_type"]


def test_simulated_source_streams_recording():
    rec = make_recording(np.random.default_rng(2), None, n=10_000)
    src = SimulatedSource(rec)
    chunks = [src.read(4_096) for _ in range(4)]
    assert [len(c) for c in chunks] == [4096, 4096, 1808, 0]
    assert np.array_equal(np.concatenate(chunks), rec.iq)
    assert src.fs == rec.fs


def test_window_labels():
    n, win, hop = 16_384, 4_096, 2_048
    mask = np.zeros(n, dtype=bool)
    mask[6_000:12_000] = True
    iq = np.arange(n).astype(complex)
    w, labels, starts = window_signal(iq, mask, win, hop)
    assert w.shape == (7, win)
    assert list(starts) == [0, 2048, 4096, 6144, 8192, 10240, 12288]
    assert list(labels) == [0, -1, 1, 1, 1, -1, 0]
    assert np.array_equal(w[2], iq[4096:8192])


def test_collect_windows_shapes():
    recs = [make_recording(np.random.default_rng(i), f) for i, f in enumerate([None, "tone"])]
    d = collect_windows(recs, 4096, 2048)
    n_win = (N_DEFAULT - 4096) // 2048 + 1
    assert d["x"].shape == (2 * n_win, 4096) and d["x"].dtype == np.complex64
    assert len(d["y"]) == len(d["family"]) == len(d["snr_db"]) == len(d["rec_id"]) == 2 * n_win
    assert set(d["family"]) == {"normal", "tone"}
    assert (d["y"][d["family"] == "normal"] == 0).all()


def test_split_family_groups_never_overlap():
    assert not set(TUNE_FAMILIES) & set(HELDOUT_FAMILIES)
    assert set(TUNE_FAMILIES) | set(HELDOUT_FAMILIES) == set(FAMILIES)
    assert None in split_families("train") and len(split_families("train")) == 1
    assert set(split_families("val")) == {None, *TUNE_FAMILIES}
    assert set(split_families("test")) == {None, *FAMILIES}


def test_split_contents():
    train = build_split("train", 4)
    val = build_split("val", 6)
    test = build_split("test", 14)
    assert {r.meta["family"] for r in train} == {"normal"}
    assert {r.meta["family"] for r in val} == {"normal", *TUNE_FAMILIES}
    assert {r.meta["family"] for r in test} == {"normal", *FAMILIES}
    assert not np.array_equal(train[0].iq, val[0].iq)
    assert not np.array_equal(val[0].iq, test[0].iq)


def test_spectrogram_tone_in_right_bin():
    fs, n = 1_000_000, 8_192
    t, f, s_db = spectrogram_db(tone(n, fs, 100_000), fs)
    assert s_db.shape == (256, len(t))
    assert f[np.argmax(s_db.mean(axis=1))] == pytest.approx(100_000, abs=fs / 256)