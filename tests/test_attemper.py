"""attemper 单元测试。"""

import numpy as np
import pytest

from tentacle.utils import attemper


def test_attemper_temperature_one_returns_same_distribution():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    out = attemper(p, 1.0)
    np.testing.assert_allclose(out, p)
    assert np.isclose(out.sum(), 1.0)


def test_attemper_output_sums_to_one():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    out = attemper(p, 0.5)
    assert out.shape == p.shape
    np.testing.assert_allclose(out, p)
    assert np.isclose(out.sum(), 1.0)


def test_attemper_illegal_positions_are_zero():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    legal = np.array([1.0, 0.0, 1.0, 0.0])
    out = attemper(p, 2.0, legal)
    assert out[1] == 0.0
    assert out[3] == 0.0
    assert np.isclose(out.sum(), 1.0)
    np.testing.assert_allclose(out[legal == 0], 0.0)


def test_attemper_legal_renormalizes_only_legal_mass():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    legal = np.array([1.0, 0.0, 1.0, 0.0])
    t = 1.0
    expected = np.zeros_like(p)
    expected[0] = 0.1 / (0.1 + 0.3)
    expected[2] = 0.3 / (0.1 + 0.3)
    out = attemper(p, t, legal)
    np.testing.assert_allclose(out, expected)


def test_attemper_low_temperature_sharpens_distribution():
    p = np.array([0.1, 0.2, 0.7])
    cold = attemper(p, 0.5)
    warm = attemper(p, 2.0)
    assert cold[2] > warm[2]


def test_attemper_raises_when_no_legal_mass():
    p = np.array([0.5, 0.5])
    legal = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="no positive probability mass"):
        attemper(p, 1.0, legal)


def test_attemper_non_positive_temperature_raises():
    with pytest.raises(AssertionError):
        attemper(np.array([1.0]), 0.0)


def test_attemper_legal_none_same_as_all_ones():
    p = np.array([0.1, 0.4, 0.2, 0.3])
    t = 0.7
    a = attemper(p, t, None)
    b = attemper(p, t, np.ones_like(p))
    np.testing.assert_allclose(a, b)


def test_attemper_unnormalized_input_renormalizes():
    """输入不必是和为 1 的分布（如计数或未归一化正数）；T=1 时等价于按比例归一化。"""
    raw = np.array([8.0, 2.0, 2.0])
    assert not np.isclose(raw.sum(), 1.0)
    out = attemper(raw, 1.0)
    expected = raw / raw.sum()
    np.testing.assert_allclose(out, expected)
    assert np.isclose(out.sum(), 1.0)
