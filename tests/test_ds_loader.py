"""DatasetLoader 单元测试。"""

from pathlib import Path

import numpy as np
import pytest

from tentacle.ds_loader import DatasetLoader


def _write_rows(path: Path, rows):
    path.write_text("\n".join(",".join(map(str, row)) for row in rows) + "\n", encoding="utf-8")


def test_load_invalid_amount_raises(tmp_path):
    path = tmp_path / "data.csv"
    _write_rows(path, [[1, 2, 3]])
    loader = DatasetLoader(str(path))
    with pytest.raises(ValueError, match="amount must be positive"):
        loader.load(0)


def test_load_empty_file_marks_wane_and_has_no_more(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    loader = DatasetLoader(str(path))
    content, has_more = loader.load(5)
    assert content.size == 0
    assert not has_more
    assert loader.is_wane


def test_load_crosses_eof_then_rewinds(tmp_path):
    path = tmp_path / "data.csv"
    _write_rows(path, [[1, 10], [2, 20], [3, 30]])
    loader = DatasetLoader(str(path))

    part1, has_more1 = loader.load(2)
    assert part1.shape == (2, 2)
    assert has_more1
    assert not loader.is_wane

    part2, has_more2 = loader.load(2)
    assert part2.shape == (2, 2)
    assert not has_more2
    assert not loader.is_wane


def test_load_detects_appended_rows_with_linecache_refresh(tmp_path):
    path = tmp_path / "grow.csv"
    _write_rows(path, [[1, 11]])
    loader = DatasetLoader(str(path))

    first, has_more_first = loader.load(1)
    np.testing.assert_allclose(first, np.array([[1.0, 11.0]]))
    assert not has_more_first

    path.write_text("1,11\n2,22\n", encoding="utf-8")
    second, _ = loader.load(1)
    np.testing.assert_allclose(second, np.array([[2.0, 22.0]]))


def test_is_wane_unsets_after_file_grows(tmp_path):
    path = tmp_path / "wane.csv"
    _write_rows(path, [[1, 1]])
    loader = DatasetLoader(str(path))

    _, _ = loader.load(5)
    assert loader.is_wane

    path.write_text("1,1\n2,2\n", encoding="utf-8")
    assert not loader.is_wane
