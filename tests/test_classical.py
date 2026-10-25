# FILE: tests/test_classical.py
# What this file does:
#   - Checks src/models/classical.py and src/data/loading.py.
#   - Both detectors: shape of the scores, higher score for shifted (anomalous)
#     data than for normal data (AUROC > 0.95 on an easy synthetic problem),
#     a clear error if score() is called before fit().
#   - Isolation Forest with the same seed gives identical scores.
#   - feature_names subset: changing a feature that was NOT selected does not
#     change the scores.
#   - load_features: removes ambiguous windows, creates the cache file on the
#     first call and returns identical data from the cache on the second.
#   Run with:  python -m pytest tests -v

import numpy as np
import pytest

from src.data.loading import load_features
from src.dsp.features import FEATURE_NAMES
from src.dsp.windowing import collect_windows
from src.eval.metrics import auroc
from src.models.classical import isolation_forest, one_class_svm
from src.sources.simulated import make_recording

DETECTORS = {"iforest": isolation_forest, "ocsvm": one_class_svm}


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((500, 13))
    normal = rng.standard_normal((300, 13))
    anomalous = rng.standard_normal((300, 13)) + 4.0
    return train, normal, anomalous


@pytest.mark.parametrize("name", DETECTORS)
def test_detector_separates_shifted_data(name):
    train, normal, anomalous = make_data()
    det = DETECTORS[name]().fit(train)
    s_n, s_a = det.score(normal), det.score(anomalous)
    assert s_n.shape == (300,) and s_a.shape == (300,)
    assert auroc(s_n, s_a) > 0.95


@pytest.mark.parametrize("name", DETECTORS)
def test_score_before_fit_raises(name):
    with pytest.raises(RuntimeError):
        DETECTORS[name]().score(np.zeros((3, 13)))


def test_isolation_forest_is_reproducible():
    train, normal, _ = make_data()
    a = isolation_forest(seed=3).fit(train).score(normal)
    b = isolation_forest(seed=3).fit(train).score(normal)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", DETECTORS)
def test_feature_subset_ignores_other_columns(name):
    train, normal, _ = make_data()
    subset = ["power_db", "flatness", "centroid"]
    det = DETECTORS[name](feature_names=subset).fit(train)
    changed = normal.copy()
    changed[:, FEATURE_NAMES.index("line4_db")] += 100.0          # not in the subset
    assert np.allclose(det.score(normal), det.score(changed))


def test_load_features_removes_ambiguous_and_caches(tmp_path):
    recs = [make_recording(np.random.default_rng(i), f) for i, f in enumerate([None, "tone"])]
    d = collect_windows(recs, 4096, 2048)
    np.savez(tmp_path / "val.npz", **d)
    first = load_features("val", tmp_path)
    assert (tmp_path / "val_features.npz").exists()
    assert len(first["y"]) == np.sum(d["y"] != -1) < len(d["y"])
    assert first["feat"].shape == (len(first["y"]), 13)
    second = load_features("val", tmp_path)
    assert all(np.array_equal(first[k], second[k]) for k in first)
