"""ReplayMemory 单元测试。"""

import numpy as np
import pytest

from tentacle.utils import ReplayMemory


def test_replay_memory_first_append_uses_index_zero():
    m = ReplayMemory(10)
    m.append("a")
    assert list(m.indexes) == [0]
    assert m.data[0] == "a"


def test_replay_memory_append_fills_sequential_indices_until_capacity():
    m = ReplayMemory(4)
    for i in range(4):
        m.append(i)
    assert list(m.indexes) == [0, 1, 2, 3]
    assert m.data == {0: 0, 1: 1, 2: 2, 3: 3}
    assert m.is_full()


def test_replay_memory_overwrite_evicts_oldest_and_reuses_index():
    m = ReplayMemory(3)
    for i in range(3):
        m.append(f"x{i}")
    assert m.data[0] == "x0" and m.data[1] == "x1" and m.data[2] == "x2"

    m.append("x3")
    assert list(m.indexes) == [1, 2, 0]
    assert m.data[0] == "x3"
    assert m.data[1] == "x1"
    assert m.data[2] == "x2"
    assert len(m.data) == 3


def test_replay_memory_sample_size_and_membership():
    m = ReplayMemory(100)
    for i in range(20):
        m.append(i)
    stored = set(m.data.values())
    for n in (1, 5, 20):
        batch = m.sample(n)
        assert len(batch) == n
        assert set(batch).issubset(stored)


def test_replay_memory_sample_all_is_permutation_of_stored():
    m = ReplayMemory(5)
    for i in range(5):
        m.append(i * 10)
    batch = m.sample(5)
    assert sorted(batch) == [0, 10, 20, 30, 40]


def test_replay_memory_sample_invalid_n_raises():
    m = ReplayMemory(5)
    m.append(1)
    with pytest.raises(AssertionError, match="brain volume too small"):
        m.sample(2)


def test_replay_memory_is_big_enough_and_clear():
    m = ReplayMemory(3)
    assert not m.is_big_enough(1)
    m.append("a")
    assert m.is_big_enough(1)
    assert not m.is_big_enough(2)
    m.clear()
    assert len(m.indexes) == 0
    assert m.data == {}
    assert not m.is_full()


def test_replay_memory_dump_roundtrip(tmp_path):
    m = ReplayMemory(4)
    for i in range(4):
        m.append(float(i))
    path = tmp_path / "mem.npz"
    m.dump(path)
    loaded = np.load(path)
    key = "arr_0" if "arr_0" in loaded.files else loaded.files[0]
    arr = loaded[key]
    np.testing.assert_allclose(arr.ravel(), np.array([0.0, 1.0, 2.0, 3.0]))
