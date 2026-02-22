"""DataFrame — two-dimensional labeled tabular data."""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .index import Index
from .series import Series


# ======================================================================
# Accessor helpers
# ======================================================================

class _DFLocIndexer:
    """Label-based indexer for DataFrame (``df.loc[row, col]``)."""

    def __init__(self, df: "DataFrame"):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = slice(None)

        # --- resolve rows ---
        if isinstance(row_key, slice):
            # label-based slicing is inclusive on both ends
            start = 0 if row_key.start is None else self._df.index.get_loc(row_key.start)
            stop = len(self._df) - 1 if row_key.stop is None else self._df.index.get_loc(row_key.stop)
            row_positions = list(range(start, stop + 1))
        elif isinstance(row_key, list):
            row_positions = self._df.index.get_locs(row_key)
        elif isinstance(row_key, (Series, np.ndarray)):
            mask = np.array(row_key._data if isinstance(row_key, Series) else row_key, dtype=bool)
            row_positions = list(np.where(mask)[0])
        else:
            row_positions = [self._df.index.get_loc(row_key)]

        # --- resolve cols ---
        if isinstance(col_key, slice) and col_key == slice(None):
            col_names = self._df.columns.tolist()
        elif isinstance(col_key, list):
            col_names = col_key
        elif isinstance(col_key, str):
            col_names = [col_key]
        else:
            col_names = self._df.columns.tolist()

        # --- single cell ---
        if len(row_positions) == 1 and len(col_names) == 1:
            val = self._df._data[col_names[0]][row_positions[0]]
            return val.item() if isinstance(val, np.generic) else val

        # --- sub-DataFrame ---
        idx = np.array(row_positions)
        new_data = {c: self._df._data[c][idx] for c in col_names}
        new_index = self._df.index[idx]
        return DataFrame(new_data, index=new_index)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = slice(None)

        row_pos = self._df.index.get_loc(row_key)
        if isinstance(col_key, str):
            self._df._data[col_key][row_pos] = value
        else:
            for c in self._df.columns:
                self._df._data[c][row_pos] = value


class _DFILocIndexer:
    """Positional indexer for DataFrame (``df.iloc[row, col]``)."""

    def __init__(self, df: "DataFrame"):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = slice(None)

        cols = self._df.columns.tolist()

        # --- resolve rows ---
        if isinstance(row_key, (int, np.integer)):
            row_positions = [int(row_key)]
        elif isinstance(row_key, slice):
            row_positions = list(range(*row_key.indices(len(self._df))))
        elif isinstance(row_key, list):
            row_positions = row_key
        else:
            row_positions = [int(row_key)]

        # --- resolve cols ---
        if isinstance(col_key, slice) and col_key == slice(None):
            col_names = cols
        elif isinstance(col_key, (int, np.integer)):
            col_names = [cols[int(col_key)]]
        elif isinstance(col_key, slice):
            col_names = cols[col_key]
        elif isinstance(col_key, list):
            col_names = [cols[i] if isinstance(i, (int, np.integer)) else i for i in col_key]
        else:
            col_names = cols

        # --- single cell ---
        if len(row_positions) == 1 and len(col_names) == 1:
            val = self._df._data[col_names[0]][row_positions[0]]
            return val.item() if isinstance(val, np.generic) else val

        idx = np.array(row_positions)
        new_data = {c: self._df._data[c][idx] for c in col_names}
        new_index = self._df.index[idx]
        return DataFrame(new_data, index=new_index)


# ======================================================================
# DataFrame
# ======================================================================

class DataFrame:
    """A two-dimensional labelled data structure (dict of Series over a shared Index)."""

    def __init__(
        self,
        data=None,
        index: Optional[Union[Index, Sequence]] = None,
        columns: Optional[Sequence[str]] = None,
    ):
        self._data: Dict[str, np.ndarray] = {}
        self._columns: Index  # Type hint for linter

        if data is None:
            data = {}

        # ---- dict of lists / arrays ----
        if isinstance(data, dict):
            col_names = columns if columns is not None else list(data.keys())
            length = None
            for c in col_names:
                arr = np.array(data[c]) if c in data else np.array([])
                self._data[c] = arr
                length = len(arr) if length is None else length

            if length is None:
                length = 0

        # ---- list of dicts (records) ----
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            col_names = columns if columns is not None else list(data[0].keys())
            for c in col_names:
                self._data[c] = np.array([row.get(c, np.nan) for row in data])
            length = len(data)

        else:
            raise TypeError(f"Unsupported data type for DataFrame: {type(data)}")

        # ---- index ----
        if index is None:
            self.index = Index(range(length))
        elif isinstance(index, Index):
            self.index = index
        else:
            self.index = Index(index)

        self._columns = Index(col_names) if not isinstance(col_names, Index) else col_names

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def columns(self) -> Index:
        return self._columns

    @columns.setter
    def columns(self, new_columns):
        if not isinstance(new_columns, Index):
            new_columns = Index(new_columns)
        if len(new_columns) != len(self._columns):
            raise ValueError("Length mismatch")
        old_cols = self._columns.tolist()
        new_cols = new_columns.tolist()
        new_data = {}
        for old_c, new_c in zip(old_cols, new_cols):
            new_data[new_c] = self._data[old_c]
        self._data = new_data
        self._columns = new_columns

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.index), len(self._columns))

    @property
    def dtypes(self) -> Series:
        return Series(
            {c: str(self._data[c].dtype) for c in self._columns},
        )

    @property
    def values(self) -> np.ndarray:
        return np.column_stack([self._data[c] for c in self._columns])

    @property
    def loc(self) -> _DFLocIndexer:
        return _DFLocIndexer(self)

    @property
    def iloc(self) -> _DFILocIndexer:
        return _DFILocIndexer(self)

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------
    # Column access
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        # boolean mask (Series or ndarray)
        if isinstance(key, Series):
            mask = np.array(key._data, dtype=bool)
            idx = np.where(mask)[0]
            new_data = {c: self._data[c][idx] for c in self._columns}
            return DataFrame(new_data, index=self.index[idx])
        if isinstance(key, np.ndarray) and key.dtype == bool:
            idx = np.where(key)[0]
            new_data = {c: self._data[c][idx] for c in self._columns}
            return DataFrame(new_data, index=self.index[idx])
        # list of columns
        if isinstance(key, list):
            new_data = {c: self._data[c] for c in key}
            return DataFrame(new_data, index=self.index.copy())
        # single column
        if key in self._data:
            return Series(self._data[key], index=self.index.copy(), name=key)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, Series):
            arr = value._data
        elif isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.full(len(self.index), value)
        self._data[key] = arr
        if key not in self._columns:
            self._columns = self._columns.append(Index([key]))

    def __getattr__(self, name: str):
        if name.startswith("_") or name in ("index", "loc", "iloc", "columns", "shape", "dtypes", "values"):
            raise AttributeError(name)
        if name in self._data:
            return self[name]
        raise AttributeError(f"'DataFrame' has no attribute '{name}'")

    def __delitem__(self, key):
        if key not in self._data:
            raise KeyError(key)
        self._data.pop(key)
        cols = [c for c in self._columns if c != key]
        self._columns = Index(cols)

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def head(self, n: int = 5) -> "DataFrame":
        return self.iloc[:n]

    def tail(self, n: int = 5) -> "DataFrame":
        return self.iloc[-n:]

    def info(self) -> str:
        lines = [f"<DataFrame>", f"Index: {len(self.index)} entries"]
        for c in self._columns:
            lines.append(f"  {c}: {self._data[c].dtype}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def _agg(self, func_name: str, **kwargs) -> Series:
        results = {}
        for c in self._columns:
            arr = self._data[c]
            try:
                arr_f = arr.astype(float)
                fn = getattr(np, f"nan{func_name}", None) or getattr(np, func_name)
                results[c] = float(fn(arr_f, **kwargs))
            except (ValueError, TypeError):
                results[c] = np.nan
        return Series(results)

    def sum(self):    return self._agg("sum")
    def mean(self):   return self._agg("mean")
    def min(self):    return self._agg("min")
    def max(self):    return self._agg("max")
    def std(self, ddof=1):  return self._agg("std", ddof=ddof)

    def describe(self) -> "DataFrame":
        result = {}
        for c in self._columns:
            s = Series(self._data[c], name=c)
            try:
                desc = s.describe()
                result[c] = desc._data
            except (ValueError, TypeError):
                continue
        if not result:
            return DataFrame()
        idx = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        return DataFrame(result, index=idx)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def apply(self, func: Callable, axis: int = 0) -> Union[Series, "DataFrame"]:
        if axis == 0:
            # apply to each column
            results = {}
            for c in self._columns:
                s = Series(self._data[c], index=self.index.copy(), name=c)
                results[c] = func(s)
            if isinstance(list(results.values())[0], Series):
                return DataFrame({k: v._data for k, v in results.items()}, index=self.index.copy())
            return Series(results)
        else:
            # apply to each row
            out = []
            for i in range(len(self)):
                row = {c: self._data[c][i] for c in self._columns}
                out.append(func(Series(row)))
            return Series(out, index=self.index.copy())

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        arr = self._data[by]
        order = np.argsort(arr)
        if not ascending:
            order = order[::-1]
        new_data = {c: self._data[c][order] for c in self._columns}
        return DataFrame(new_data, index=self.index[order])

    def drop(self, labels=None, columns=None) -> "DataFrame":
        new_data = {c: self._data[c].copy() for c in self._columns}
        new_index = self.index.copy()

        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            for c in columns:
                del new_data[c]
            return DataFrame(new_data, index=new_index)

        if labels is not None:
            if not isinstance(labels, list):
                labels = [labels]
            positions = self.index.get_locs(labels)
            mask = np.ones(len(self.index), dtype=bool)
            mask[positions] = False
            new_data = {c: self._data[c][mask] for c in self._columns}
            return DataFrame(new_data, index=self.index[mask])

        return DataFrame(new_data, index=new_index)

    def rename(self, columns: Optional[Dict[str, str]] = None) -> "DataFrame":
        if columns is None:
            return self.copy()
        new_data = {}
        for c in self._columns:
            new_name = columns.get(c, c)
            new_data[new_name] = self._data[c].copy()
        return DataFrame(new_data, index=self.index.copy())

    def reset_index(self, drop: bool = False) -> "DataFrame":
        new_data = {}
        if not drop:
            new_data["index"] = np.array(self.index.tolist())
        for c in self._columns:
            new_data[c] = self._data[c].copy()
        return DataFrame(new_data, index=Index(range(len(self.index))))

    def copy(self) -> "DataFrame":
        new_data = {c: self._data[c].copy() for c in self._columns}
        return DataFrame(new_data, index=self.index.copy())

    def iterrows(self):
        for i in range(len(self)):
            label = self.index[i]
            row = Series({c: self._data[c][i] for c in self._columns})
            yield label, row

    def astype(self, dtype) -> "DataFrame":
        new_data = {c: self._data[c].astype(dtype) for c in self._columns}
        return DataFrame(new_data, index=self.index.copy())

    def fillna(self, value) -> "DataFrame":
        new_data = {}
        for c in self._columns:
            arr = self._data[c].copy()
            try:
                mask = np.isnan(arr.astype(float))
                arr = arr.astype(float)
                arr[mask] = value
            except (ValueError, TypeError):
                pass
            new_data[c] = arr
        return DataFrame(new_data, index=self.index.copy())

    def dropna(self) -> "DataFrame":
        mask = np.ones(len(self), dtype=bool)
        for c in self._columns:
            try:
                col_data = self._data[c]
                # Use np.logical_and to satisfy linter type checking
                valid = ~np.isnan(col_data.astype(float))
                mask = np.logical_and(mask, valid)
            except (ValueError, TypeError):
                pass
        idx = np.where(mask)[0]
        new_data = {c: self._data[c][idx] for c in self._columns}
        return DataFrame(new_data, index=self.index[idx])

    def drop_duplicates(
        self,
        subset: Optional[Union[str, List[str]]] = None,
        keep: str = "first",
    ) -> "DataFrame":
        """Remove duplicate rows.

        Parameters
        ----------
        subset : str, list of str, or None (default None)
            Column label(s) to consider for identifying duplicates.
            ``None`` means all columns.
        keep : {'first', 'last', False}, default 'first'
            - 'first' : keep the first occurrence.
            - 'last'  : keep the last occurrence.
            - False   : drop **all** duplicated rows.
        """
        if subset is None:
            cols = self._columns.tolist()
        elif isinstance(subset, str):
            cols = [subset]
        else:
            cols = list(subset)

        n = len(self)
        mask = np.ones(n, dtype=bool)

        def _row_key(i: int):
            return tuple(
                v.item() if isinstance(v, np.generic) else v
                for v in (self._data[c][i] for c in cols)
            )

        seen: dict = {}

        if keep == "first":
            for i in range(n):
                key = _row_key(i)
                if key in seen:
                    mask[i] = False
                else:
                    seen[key] = i
        elif keep == "last":
            for i in range(n - 1, -1, -1):
                key = _row_key(i)
                if key in seen:
                    mask[i] = False
                else:
                    seen[key] = i
        elif keep is False:
            counts: dict = {}
            for i in range(n):
                key = _row_key(i)
                counts.setdefault(key, []).append(i)
            for positions in counts.values():
                if len(positions) > 1:
                    for p in positions:
                        mask[p] = False
        else:
            raise ValueError(f"keep must be 'first', 'last', or False, got {keep!r}")

        idx = np.where(mask)[0]
        new_data = {c: self._data[c][idx] for c in self._columns}
        return DataFrame(new_data, index=self.index[idx])

    # ------------------------------------------------------------------
    # GroupBy delegation
    # ------------------------------------------------------------------

    def groupby(self, by: Union[str, List[str]]) -> "GroupBy":
        from .groupby import GroupBy
        return GroupBy(self, by)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_csv(self, filepath: str, index: bool = True) -> None:
        from .io import _to_csv
        _to_csv(self, filepath, index=index)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if len(self._columns) == 0:
            return "Empty DataFrame"

        n = len(self)
        all_cols = self._columns.tolist()
        num_cols = len(all_cols)
        labels = self.index.tolist()

        # ── Display settings ──────────────────────────────────────────
        max_rows = 10        # show head/tail of 5 each when exceeded
        head_rows = 5
        tail_rows = 5
        max_cols = 10        # show first/last 5 each when exceeded
        head_cols = 5
        tail_cols = 5
        max_cell_width = 20  # truncate long cell values

        # ── Determine which rows and columns to display ───────────────
        truncate_rows = n > max_rows
        truncate_cols = num_cols > max_cols

        if truncate_rows:
            row_indices = list(range(head_rows)) + list(range(n - tail_rows, n))
        else:
            row_indices = list(range(n))

        if truncate_cols:
            display_cols = all_cols[:head_cols] + all_cols[-tail_cols:]
        else:
            display_cols = all_cols

        # ── Helpers ───────────────────────────────────────────────────
        def _format_val(v):
            """Convert a raw array element to a display string."""
            v = v.item() if isinstance(v, np.generic) else v
            s = str(v)
            if len(s) > max_cell_width:
                s = s[: max_cell_width - 1] + "…"
            return s

        def _is_numeric_col(col_name):
            """Check if the column holds numeric data."""
            return np.issubdtype(self._data[col_name].dtype, np.number)

        # ── Build cell grid (list of lists) ───────────────────────────
        # Each "grid row" is a list of cell strings.
        header_cells = []
        col_numeric = []
        for c in display_cols:
            h = str(c)
            if len(h) > max_cell_width:
                h = h[: max_cell_width - 1] + "…"
            header_cells.append(h)
            col_numeric.append(_is_numeric_col(c))

        if truncate_cols:
            header_cells.insert(head_cols, "···")
            col_numeric.insert(head_cols, False)

        # Index labels for displayed rows
        disp_labels = [str(labels[i]) for i in row_indices]

        data_rows = []
        for i in row_indices:
            row_cells = []
            for c in display_cols:
                row_cells.append(_format_val(self._data[c][i]))
            if truncate_cols:
                row_cells.insert(head_cols, "···")
            data_rows.append(row_cells)

        # Insert an ellipsis row when rows are truncated
        if truncate_rows:
            ellipsis_row = ["⋮" for _ in header_cells]
            data_rows.insert(head_rows, ellipsis_row)
            disp_labels.insert(head_rows, "⋮")

        # ── Compute column widths ─────────────────────────────────────
        num_display_cols = len(header_cells)
        col_widths = [len(header_cells[j]) for j in range(num_display_cols)]
        for row_cells in data_rows:
            for j, cell in enumerate(row_cells):
                col_widths[j] = max(col_widths[j], len(cell))

        idx_width = max((len(l) for l in disp_labels), default=0)

        # ── Render ────────────────────────────────────────────────────
        def _align(val, width, numeric):
            return val.rjust(width) if numeric else val.ljust(width)

        # Header line
        header_line = " " * idx_width + "  " + "  ".join(
            _align(header_cells[j], col_widths[j], col_numeric[j])
            for j in range(num_display_cols)
        )
        # Separator line
        sep_line = "─" * idx_width + "──" + "──".join(
            "─" * col_widths[j] for j in range(num_display_cols)
        )

        lines = [header_line, sep_line]

        for idx_label, row_cells in zip(disp_labels, data_rows):
            row_str = idx_label.ljust(idx_width) + "  " + "  ".join(
                _align(row_cells[j], col_widths[j], col_numeric[j])
                for j in range(num_display_cols)
            )
            lines.append(row_str)

        # Footer
        lines.append("")
        lines.append(f"[{n} rows × {num_cols} columns]")

        return "\n".join(lines)
