# FILE: tests/test_seeds.py
# What this file does:
#   - Checks mean_std on known numbers and that one seed gives std 0.
#   Run with:  python -m pytest tests -v

import pytest

from src.eval.seeds import mean_std


def test_mean_std_known_values():
    out = mean_std([{"a": 1.0, "b": 5.0}, {"a": 3.0, "b": 5.0}])
    assert out["a"] == pytest.approx((2.0, 2 ** 0.5))
    assert out["b"] == pytest.approx((5.0, 0.0))


def test_single_seed_has_zero_std():
    assert mean_std([{"a": 0.7}])["a"] == (0.7, 0.0)