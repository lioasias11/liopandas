"""Tests for liopandas.Series"""

import numpy as np
import pytest
from liopandas import Series


class TestConstruction:
    def test_from_list(self):
        s = Series([1, 2, 3])
        assert len(s) == 3
        assert list(s) == [1, 2, 3]

    def test_from_dict(self):
        s = Series({"a": 10, "b": 20})
        assert s["a"] == 10
        assert s["b"] == 20

    def test_with_custom_index(self):
        s = Series([10, 20], index=["x", "y"], name="vals")
        assert s["x"] == 10
        assert s.name == "vals"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            Series([1, 2, 3], index=["a", "b"])


class TestAccessors:
    def test_loc(self):
        s = Series([10, 20, 30], index=["a", "b", "c"])
        assert s.loc["b"] == 20

    def test_loc_list(self):
        s = Series([10, 20, 30], index=["a", "b", "c"])
        result = s.loc[["a", "c"]]
        assert list(result) == [10, 30]

    def test_iloc(self):
        s = Series([10, 20, 30])
        assert s.iloc[1] == 20

    def test_iloc_slice(self):
        s = Series([10, 20, 30, 40])
        result = s.iloc[1:3]
        assert list(result) == [20, 30]


class TestArithmetic:
    def test_add(self):
        s = Series([1, 2, 3])
        result = s + 10
        assert list(result) == [11, 12, 13]

    def test_sub_series(self):
        a = Series([10, 20, 30])
        b = Series([1, 2, 3])
        result = a - b
        assert list(result) == [9, 18, 27]

    def test_mul(self):
        s = Series([2, 3, 4])
        result = s * 2
        assert list(result) == [4, 6, 8]

    def test_div(self):
        s = Series([10, 20, 30])
        result = s / 10
        assert list(result) == [1.0, 2.0, 3.0]


class TestComparisons:
    def test_gt(self):
        s = Series([1, 5, 3])
        mask = s > 2
        assert list(mask) == [False, True, True]

    def test_eq(self):
        s = Series([1, 2, 2, 3])
        mask = s == 2
        assert list(mask) == [False, True, True, False]


class TestBooleanIndexing:
    def test_filter(self):
        s = Series([10, 20, 30, 40])
        result = s[s > 15]
        assert list(result) == [20, 30, 40]


class TestAggregations:
    def test_sum(self):
        assert Series([1, 2, 3]).sum() == 6.0

    def test_mean(self):
        assert Series([2, 4, 6]).mean() == 4.0

    def test_min_max(self):
        s = Series([3, 1, 2])
        assert s.min() == 1.0
        assert s.max() == 3.0

    def test_std(self):
        s = Series([2, 4, 4, 4, 5, 5, 7, 9])
        assert round(s.std(), 4) == 2.1381

    def test_describe(self):
        s = Series([1, 2, 3, 4, 5])
        desc = s.describe()
        assert desc["count"] == 5
        assert desc["mean"] == 3.0


class TestUtilities:
    def test_head_tail(self):
        s = Series(range(10))
        assert len(s.head(3)) == 3
        assert len(s.tail(2)) == 2

    def test_apply(self):
        s = Series([1, 2, 3])
        result = s.apply(lambda x: x ** 2)
        assert list(result) == [1, 4, 9]

    def test_value_counts(self):
        s = Series(["a", "b", "a", "a", "b"])
        vc = s.value_counts()
        assert vc["a"] == 3
        assert vc["b"] == 2

    def test_sort_values(self):
        s = Series([3, 1, 2])
        assert list(s.sort_values()) == [1, 2, 3]
        assert list(s.sort_values(ascending=False)) == [3, 2, 1]

    def test_unique(self):
        s = Series([1, 2, 2, 3])
        assert set(s.unique()) == {1, 2, 3}

    def test_nunique(self):
        assert Series([1, 1, 2]).nunique() == 2

    def test_isna_fillna_dropna(self):
        s = Series([1.0, np.nan, 3.0])
        assert s.isna().sum() == 1.0
        filled = s.fillna(0)
        assert list(filled) == [1.0, 0.0, 3.0]
        dropped = s.dropna()
        assert len(dropped) == 2
