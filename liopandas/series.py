"""Series — one-dimensional labeled array."""

from __future__ import annotations

import math
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .index import Index


class _LocIndexer:
    """Label-based indexer for Series (``s.loc[...]``)."""

    def __init__(self, series: "Series"):
        self._s = series

    def __getitem__(self, key):
        if isinstance(key, list):
            positions = self._s.index.get_locs(key)
            return Series(
                self._s._data[positions],
                index=self._s.index[positions],
                name=self._s.name,
            )
        pos = self._s.index.get_loc(key)
        val = self._s._data[pos]
        return val.item() if isinstance(val, np.generic) else val

    def __setitem__(self, key, value):
        pos = self._s.index.get_loc(key)
        self._s._data[pos] = value


class _ILocIndexer:
    """Positional indexer for Series (``s.iloc[...]``)."""

    def __init__(self, series: "Series"):
        self._s = series

    def __getitem__(self, key):
        if isinstance(key, list):
            idx = np.array(key)
            return Series(
                self._s._data[idx],
                index=self._s.index[idx],
                name=self._s.name,
            )
        if isinstance(key, slice):
            return Series(
                self._s._data[key],
                index=self._s.index[key],
                name=self._s.name,
            )
        val = self._s._data[key]
        return val.item() if isinstance(val, np.generic) else val

    def __setitem__(self, key, value):
        self._s._data[key] = value


# ======================================================================
# Series
# ======================================================================

class Series:
    """A one-dimensional labeled array backed by a NumPy array."""

    def __init__(
        self,
        data=None,
        index: Optional[Union[Index, Sequence]] = None,
        name: Optional[str] = None,
        dtype=None,
    ):
        # ---- handle dict input ----
        if isinstance(data, dict):
            keys = list(data.keys())
            vals = list(data.values())
            if index is None:
                index = keys
            data = vals

        # ---- coerce data to ndarray ----
        if data is None:
            data = []
        if isinstance(data, np.ndarray):
            self._data = data.copy() if dtype is None else data.astype(dtype)
        else:
            self._data = np.array(list(data) if not isinstance(data, list) else data,
                                  dtype=dtype)

        # ---- coerce index ----
        if index is None:
            self.index = Index(range(len(self._data)))
        elif isinstance(index, Index):
            self.index = index
        else:
            self.index = Index(index)

        if len(self.index) != len(self._data):
            raise ValueError(
                f"Length of index ({len(self.index)}) does not match "
                f"length of data ({len(self._data)})"
            )

        self.name = name

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    @property
    def iloc(self) -> _ILocIndexer:
        return _ILocIndexer(self)

    @property
    def values(self) -> np.ndarray:
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for v in self._data:
            yield v.item() if isinstance(v, np.generic) else v

    def __contains__(self, item) -> bool:
        return item in self._data

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        # boolean mask
        if isinstance(key, (Series,)):
            mask = np.array(key._data, dtype=bool)
            return Series(self._data[mask], index=self.index[mask], name=self.name)
        if isinstance(key, np.ndarray) and key.dtype == bool:
            return Series(self._data[key], index=self.index[key], name=self.name)
        if isinstance(key, list) and len(key) > 0 and isinstance(key[0], bool):
            mask = np.array(key, dtype=bool)
            return Series(self._data[mask], index=self.index[mask], name=self.name)
        # label-based fallback
        if isinstance(key, list):
            positions = self.index.get_locs(key)
            return Series(
                self._data[positions],
                index=self.index[positions],
                name=self.name,
            )
        pos = self.index.get_loc(key)
        val = self._data[pos]
        return val.item() if isinstance(val, np.generic) else val

    def __setitem__(self, key, value):
        if isinstance(key, (Series,)):
            mask = np.array(key._data, dtype=bool)
            self._data[mask] = value
            return
        if isinstance(key, np.ndarray) and key.dtype == bool:
            self._data[key] = value
            return
        pos = self.index.get_loc(key)
        self._data[pos] = value

    # ------------------------------------------------------------------
    # Arithmetic (element-wise, returns new Series)
    # ------------------------------------------------------------------

    def _arith(self, other, op):
        if isinstance(other, Series):
            other = other._data
        result = op(self._data, other)
        return Series(result, index=self.index.copy(), name=self.name)

    def __add__(self, other):      return self._arith(other, np.add)
    def __radd__(self, other):     return self._arith(other, np.add)
    def __sub__(self, other):      return self._arith(other, np.subtract)
    def __rsub__(self, other):     return self._arith(other, lambda a, b: np.subtract(b, a))
    def __mul__(self, other):      return self._arith(other, np.multiply)
    def __rmul__(self, other):     return self._arith(other, np.multiply)
    def __truediv__(self, other):  return self._arith(other, np.true_divide)
    def __rtruediv__(self, other): return self._arith(other, lambda a, b: np.true_divide(b, a))
    def __floordiv__(self, other): return self._arith(other, np.floor_divide)
    def __mod__(self, other):      return self._arith(other, np.mod)
    def __pow__(self, other):      return self._arith(other, np.power)

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def __eq__(self, other):  return self._arith(other, np.equal)
    def __ne__(self, other):  return self._arith(other, np.not_equal)
    def __lt__(self, other):  return self._arith(other, np.less)
    def __le__(self, other):  return self._arith(other, np.less_equal)
    def __gt__(self, other):  return self._arith(other, np.greater)
    def __ge__(self, other):  return self._arith(other, np.greater_equal)

    # Boolean operators
    def __and__(self, other): return self._arith(other, np.logical_and)
    def __or__(self, other):  return self._arith(other, np.logical_or)
    def __invert__(self):
        return Series(~self._data, index=self.index.copy(), name=self.name)

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def sum(self):   return float(np.nansum(self._data))
    def mean(self):  return float(np.nanmean(self._data))
    def min(self):   return float(np.nanmin(self._data))
    def max(self):   return float(np.nanmax(self._data))
    def std(self, ddof=1):  return float(np.nanstd(self._data, ddof=ddof))
    def median(self): return float(np.nanmedian(self._data))
    def count(self): return int(np.count_nonzero(~np.isnan(self._data.astype(float))))

    def describe(self) -> "Series":
        stats = {
            "count": self.count(),
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "25%": float(np.nanpercentile(self._data.astype(float), 25)),
            "50%": self.median(),
            "75%": float(np.nanpercentile(self._data.astype(float), 75)),
            "max": self.max(),
        }
        return Series(stats, name=self.name)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def head(self, n: int = 5) -> "Series":
        return self.iloc[:n]

    def tail(self, n: int = 5) -> "Series":
        return self.iloc[-n:]

    def apply(self, func: Callable) -> "Series":
        result = np.array([func(x.item() if isinstance(x, np.generic) else x) for x in self._data])
        return Series(result, index=self.index.copy(), name=self.name)

    def map(self, mapping: Union[dict, Callable]) -> "Series":
        if callable(mapping):
            return self.apply(mapping)
        result = np.array([mapping.get(x.item() if isinstance(x, np.generic) else x, np.nan) for x in self._data])
        return Series(result, index=self.index.copy(), name=self.name)

    def value_counts(self) -> "Series":
        unique, counts = np.unique(self._data, return_counts=True)
        # Use np.negative to satisfy linter that doesn't recognize numpy unary minus
        order = np.argsort(np.negative(counts))
        return Series(counts[order], index=Index(unique[order]), name=self.name)

    def sort_values(self, ascending: bool = True) -> "Series":
        order = np.argsort(self._data)
        if not ascending:
            order = order[::-1]
        return Series(self._data[order], index=self.index[order], name=self.name)

    def reset_index(self, drop: bool = False) -> "Series":
        if drop:
            return Series(self._data.copy(), name=self.name)
        return Series(self._data.copy(), index=self.index.reset(), name=self.name)

    def copy(self) -> "Series":
        return Series(self._data.copy(), index=self.index.copy(), name=self.name)

    def astype(self, dtype) -> "Series":
        return Series(self._data.astype(dtype), index=self.index.copy(), name=self.name)

    def isna(self) -> "Series":
        try:
            mask = np.isnan(self._data.astype(float))
        except (ValueError, TypeError):
            mask = np.array([x is None for x in self._data])
        return Series(mask, index=self.index.copy(), name=self.name)

    def notna(self) -> "Series":
        return ~self.isna()

    def fillna(self, value) -> "Series":
        data = self._data.copy()
        try:
            mask = np.isnan(data.astype(float))
            data = data.astype(float)
            data[mask] = value
        except (ValueError, TypeError):
            data = np.array([value if x is None else x for x in data])
        return Series(data, index=self.index.copy(), name=self.name)

    def dropna(self) -> "Series":
        mask = self.notna()._data
        return Series(self._data[mask], index=self.index[mask], name=self.name)

    def drop_duplicates(self, keep: str = "first") -> "Series":
        """Remove duplicate values.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            - 'first' : keep the first occurrence of each duplicate.
            - 'last'  : keep the last occurrence of each duplicate.
            - False   : drop **all** occurrences of duplicated values.
        """
        n = len(self._data)
        mask = np.ones(n, dtype=bool)
        seen: dict = {}

        if keep == "first":
            for i in range(n):
                val = self._data[i]
                key = val.item() if isinstance(val, np.generic) else val
                if key in seen:
                    mask[i] = False
                else:
                    seen[key] = i
        elif keep == "last":
            for i in range(n - 1, -1, -1):
                val = self._data[i]
                key = val.item() if isinstance(val, np.generic) else val
                if key in seen:
                    mask[i] = False
                else:
                    seen[key] = i
        elif keep is False:
            counts: dict = {}
            for i in range(n):
                val = self._data[i]
                key = val.item() if isinstance(val, np.generic) else val
                counts.setdefault(key, []).append(i)
            for positions in counts.values():
                if len(positions) > 1:
                    for p in positions:
                        mask[p] = False
        else:
            raise ValueError(f"keep must be 'first', 'last', or False, got {keep!r}")

        return Series(self._data[mask], index=self.index[mask], name=self.name)

    def to_numpy(
        self,
        dtype: Optional[Any] = None,
        copy: bool = False,
        na_value: Any = None,
    ) -> np.ndarray:
        """Return a NumPy ndarray representing the values in this Series.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array.
        na_value : Any, optional
            The value to use for missing values. The default is None, which
            means that no replacement will be done.
        """
        data = self._data
        if na_value is not None:
            data = data.copy()
            try:
                mask = np.isnan(data.astype(float))
                data = data.astype(float)
                data[mask] = na_value
            except (ValueError, TypeError):
                data = np.array([na_value if x is None else x for x in data])

        if dtype is not None:
            data = data.astype(dtype)

        if copy:
            return data.copy()
        return data

    def unique(self) -> np.ndarray:
        return np.unique(self._data)

    def nunique(self) -> int:
        return len(self.unique())

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines: List[str] = []
        max_show = 30
        labels = self.index.tolist()
        n = len(self._data)
        show_n = min(n, max_show)
        max_label_len = max((len(str(l)) for l in labels[:show_n]), default=0)
        for i in range(show_n):
            lbl = str(labels[i]).ljust(max_label_len)
            val = self._data[i]
            val = val.item() if isinstance(val, np.generic) else val
            lines.append(f"{lbl}    {val}")
        if n > max_show:
            lines.append(f"... ({n} items total)")
        meta = []
        if self.name:
            meta.append(f"Name: {self.name}")
        meta.append(f"dtype: {self.dtype}")
        lines.append(", ".join(meta))
        return "\n".join(lines)
