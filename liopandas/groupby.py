"""GroupBy — split-apply-combine on DataFrame."""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Union

from .index import Index


class GroupBy:
    """Returned by ``DataFrame.groupby()``. Supports aggregation per group."""

    def __init__(self, df, by: Union[str, List[str]]):
        from .dataframe import DataFrame

        self._df = df
        self._by = [by] if isinstance(by, str) else list(by)
        self._groups: Dict[tuple, List[int]] = {}

        # Build group mapping
        keys_arrays = [df._data[c] for c in self._by]
        for i in range(len(df)):
            key = tuple(
                arr[i].item() if isinstance(arr[i], np.generic) else arr[i]
                for arr in keys_arrays
            )
            self._groups.setdefault(key, []).append(i)

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def agg(self, func: Union[str, Callable]) -> "DataFrame":
        from .dataframe import DataFrame
        from .series import Series

        value_cols = [c for c in self._df._columns if c not in self._by]

        result_data = {c: [] for c in self._by}
        for c in value_cols:
            result_data[c] = []

        for key, idxs in self._groups.items():
            for k_col, k_val in zip(self._by, key):
                result_data[k_col].append(k_val)

            positions = np.array(idxs)
            for c in value_cols:
                subset = self._df._data[c][positions]
                try:
                    subset = subset.astype(float)
                except (ValueError, TypeError):
                    result_data[c].append(np.nan)
                    continue

                if callable(func):
                    result_data[c].append(func(subset))
                elif func == "sum":
                    result_data[c].append(float(np.nansum(subset)))
                elif func == "mean":
                    result_data[c].append(float(np.nanmean(subset)))
                elif func == "min":
                    result_data[c].append(float(np.nanmin(subset)))
                elif func == "max":
                    result_data[c].append(float(np.nanmax(subset)))
                elif func == "count":
                    result_data[c].append(int(np.count_nonzero(~np.isnan(subset))))
                elif func == "std":
                    result_data[c].append(float(np.nanstd(subset, ddof=1)))
                else:
                    raise ValueError(f"Unknown aggregation: {func}")

        return DataFrame({c: np.array(v) for c, v in result_data.items()})

    # ------------------------------------------------------------------
    # Shorthand methods
    # ------------------------------------------------------------------

    def sum(self):   return self.agg("sum")
    def mean(self):  return self.agg("mean")
    def min(self):   return self.agg("min")
    def max(self):   return self.agg("max")
    def count(self): return self.agg("count")
    def std(self):   return self.agg("std")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<GroupBy [{', '.join(self._by)}] — {len(self._groups)} groups>"
