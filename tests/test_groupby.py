"""Tests for liopandas.GroupBy"""

import pytest
from liopandas import DataFrame


class TestGroupBy:
    def setup_method(self):
        self.df = DataFrame({
            "team": ["A", "A", "B", "B", "B"],
            "score": [10, 20, 30, 40, 50],
            "bonus": [1, 2, 3, 4, 5],
        })

    def test_repr(self):
        g = self.df.groupby("team")
        assert "2 groups" in repr(g)

    def test_sum(self):
        result = self.df.groupby("team").sum()
        # find team A row
        a_mask = result["team"] == "A"
        a_row = result[a_mask]
        assert list(a_row["score"])[0] == 30.0

    def test_mean(self):
        result = self.df.groupby("team").mean()
        b_mask = result["team"] == "B"
        b_row = result[b_mask]
        assert list(b_row["score"])[0] == 40.0

    def test_count(self):
        result = self.df.groupby("team").count()
        a_mask = result["team"] == "A"
        a_row = result[a_mask]
        assert list(a_row["score"])[0] == 2

    def test_min_max(self):
        result_min = self.df.groupby("team").min()
        result_max = self.df.groupby("team").max()
        b_mask_min = result_min["team"] == "B"
        b_mask_max = result_max["team"] == "B"
        assert list(result_min[b_mask_min]["score"])[0] == 30.0
        assert list(result_max[b_mask_max]["score"])[0] == 50.0

    def test_agg_callable(self):
        import numpy as np
        result = self.df.groupby("team").agg(np.sum)
        a_mask = result["team"] == "A"
        assert list(result[a_mask]["score"])[0] == 30.0
