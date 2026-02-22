"""Merge / Join — SQL-style joins on DataFrames."""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Union

from .index import Index


def merge(
    left,
    right,
    on: Optional[Union[str, List[str]]] = None,
    how: str = "inner",
) -> "DataFrame":
    """
    Merge two DataFrames on one or more key columns.

    Parameters
    ----------
    left, right : DataFrame
    on : str or list of str — column(s) to join on (must exist in both)
    how : 'inner', 'left', 'right', 'outer'
    """
    from .dataframe import DataFrame

    if on is None:
        # auto-detect shared columns
        left_cols = set(left._columns.tolist())
        right_cols = set(right._columns.tolist())
        on = list(left_cols & right_cols)
        if not on:
            raise ValueError("No common columns to merge on.")
    if isinstance(on, str):
        on = [on]

    # Build right-side index: key -> list of row positions
    right_map: dict = {}
    for i in range(len(right)):
        key = tuple(
            right._data[c][i].item() if isinstance(right._data[c][i], np.generic)
            else right._data[c][i]
            for c in on
        )
        right_map.setdefault(key, []).append(i)

    left_indices: List[int] = []
    right_indices: List[Optional[int]] = []
    matched_right: set = set()

    for i in range(len(left)):
        key = tuple(
            left._data[c][i].item() if isinstance(left._data[c][i], np.generic)
            else left._data[c][i]
            for c in on
        )
        matches = right_map.get(key, [])
        if matches:
            for j in matches:
                left_indices.append(i)
                right_indices.append(j)
                matched_right.add(j)
        elif how in ("left", "outer"):
            left_indices.append(i)
            right_indices.append(None)

    # right/outer: add unmatched right rows
    if how in ("right", "outer"):
        for j in range(len(right)):
            if j not in matched_right:
                left_indices.append(None)           # type: ignore[arg-type]
                right_indices.append(j)

    # Build result columns
    left_cols = left._columns.tolist()
    right_only_cols = [c for c in right._columns.tolist() if c not in on]

    result: dict = {c: [] for c in left_cols + right_only_cols}

    for li, ri in zip(left_indices, right_indices):
        for c in left_cols:
            if li is not None:
                val = left._data[c][li]
                val = val.item() if isinstance(val, np.generic) else val
            else:
                # right-only row: use right key cols, NaN for left-only cols
                if c in on and ri is not None:
                    val = right._data[c][ri]
                    val = val.item() if isinstance(val, np.generic) else val
                else:
                    val = np.nan
            result[c].append(val)

        for c in right_only_cols:
            if ri is not None:
                val = right._data[c][ri]
                val = val.item() if isinstance(val, np.generic) else val
            else:
                val = np.nan
            result[c].append(val)

    result_arrays = {c: np.array(v) for c, v in result.items()}
    return DataFrame(result_arrays)
