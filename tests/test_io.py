"""Tests for liopandas CSV I/O."""

import os
import tempfile
import pytest
from liopandas import DataFrame, read_csv


class TestCSVRoundTrip:
    def test_write_read(self):
        df = DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            df.to_csv(path, index=False)
            df2 = read_csv(path)
            assert df2.shape == (2, 2)
            assert list(df2["age"]) == [30, 25]
            assert list(df2["name"]) == ["Alice", "Bob"]
        finally:
            os.unlink(path)

    def test_write_read_with_index(self):
        df = DataFrame({"val": [10, 20, 30]}, index=["a", "b", "c"])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            df.to_csv(path, index=True)
            df2 = read_csv(path, index_col=0)
            assert len(df2) == 3
            assert list(df2["val"]) == [10, 20, 30]
        finally:
            os.unlink(path)
