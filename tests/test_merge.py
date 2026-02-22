"""Tests for liopandas.merge"""

import pytest
from liopandas import DataFrame, merge


class TestMerge:
    def setup_method(self):
        self.left = DataFrame({"key": [1, 2, 3], "val_l": ["a", "b", "c"]})
        self.right = DataFrame({"key": [2, 3, 4], "val_r": ["x", "y", "z"]})

    def test_inner(self):
        result = merge(self.left, self.right, on="key", how="inner")
        assert len(result) == 2
        assert set(result["key"]) == {2, 3}

    def test_left(self):
        result = merge(self.left, self.right, on="key", how="left")
        assert len(result) == 3

    def test_right(self):
        result = merge(self.left, self.right, on="key", how="right")
        assert len(result) == 3

    def test_outer(self):
        result = merge(self.left, self.right, on="key", how="outer")
        assert len(result) == 4

    def test_auto_detect_key(self):
        result = merge(self.left, self.right, how="inner")
        assert len(result) == 2
