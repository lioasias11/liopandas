"""Tests for liopandas.DataFrame"""

import numpy as np
import pytest
from liopandas import DataFrame, Series


class TestConstruction:
    def test_from_dict(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        assert df.shape == (2, 2)

    def test_from_records(self):
        df = DataFrame([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        assert df.shape == (2, 2)
        assert list(df["x"]) == [1, 3]

    def test_with_index(self):
        df = DataFrame({"a": [10, 20]}, index=["r1", "r2"])
        assert df.loc["r1", "a"] == 10


class TestColumnAccess:
    def test_bracket(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        s = df["a"]
        assert isinstance(s, Series)
        assert list(s) == [1, 2]

    def test_attr(self):
        df = DataFrame({"col": [10, 20]})
        assert list(df.col) == [10, 20]

    def test_set_column(self):
        df = DataFrame({"a": [1, 2]})
        df["b"] = [10, 20]
        assert list(df["b"]) == [10, 20]

    def test_delete_column(self):
        df = DataFrame({"a": [1], "b": [2]})
        del df["b"]
        assert "b" not in df.columns

    def test_multi_column(self):
        df = DataFrame({"a": [1], "b": [2], "c": [3]})
        sub = df[["a", "c"]]
        assert sub.shape == (1, 2)


class TestLocIloc:
    def test_loc_single(self):
        df = DataFrame({"a": [10, 20, 30]}, index=["x", "y", "z"])
        assert df.loc["y", "a"] == 20

    def test_iloc_single(self):
        df = DataFrame({"a": [10, 20, 30]})
        assert df.iloc[1, 0] == 20

    def test_iloc_slice(self):
        df = DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
        sub = df.iloc[1:3]
        assert sub.shape == (2, 2)


class TestBooleanFiltering:
    def test_filter(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df[df["a"] > 1]
        assert result.shape == (2, 2)
        assert list(result["a"]) == [2, 3]


class TestAggregations:
    def test_sum(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        s = df.sum()
        assert s["a"] == 6.0
        assert s["b"] == 15.0

    def test_mean(self):
        df = DataFrame({"a": [2, 4]})
        assert df.mean()["a"] == 3.0

    def test_describe(self):
        df = DataFrame({"a": [1, 2, 3, 4, 5]})
        desc = df.describe()
        assert desc.loc["count", "a"] == 5.0


class TestTransforms:
    def test_sort_values(self):
        df = DataFrame({"a": [3, 1, 2], "b": ["x", "y", "z"]})
        sorted_df = df.sort_values("a")
        assert list(sorted_df["a"]) == [1, 2, 3]

    def test_rename(self):
        df = DataFrame({"old": [1, 2]})
        df2 = df.rename(columns={"old": "new"})
        assert "new" in df2.columns

    def test_drop_columns(self):
        df = DataFrame({"a": [1], "b": [2], "c": [3]})
        df2 = df.drop(columns=["b"])
        assert list(df2.columns) == ["a", "c"]

    def test_reset_index(self):
        df = DataFrame({"a": [10, 20]}, index=["x", "y"])
        df2 = df.reset_index()
        assert "index" in df2.columns

    def test_copy(self):
        df = DataFrame({"a": [1, 2]})
        df2 = df.copy()
        df2["a"] = [99, 99]
        assert list(df["a"]) == [1, 2]  # original unchanged

    def test_fillna(self):
        df = DataFrame({"a": [1.0, np.nan, 3.0]})
        df2 = df.fillna(0)
        assert list(df2["a"]) == [1.0, 0.0, 3.0]

    def test_dropna(self):
        df = DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = df.dropna()
        assert len(df2) == 2

    def test_iterrows(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        rows = list(df.iterrows())
        assert len(rows) == 2
        assert rows[0][1]["a"] == 1
