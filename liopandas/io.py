"""I/O utilities — CSV read / write."""

from __future__ import annotations

import csv
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .dataframe import DataFrame
    from .series import Series


def read_csv(filepath: str, index_col: Optional[int] = None) -> "DataFrame":
    """
    Read a CSV file into a DataFrame.

    Numeric columns are auto-detected and converted to float.
    """
    from .dataframe import DataFrame
    from .index import Index

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        return DataFrame({h: np.array([]) for h in header})

    # Separate index column if requested
    if index_col is not None:
        idx_name = header.pop(index_col)
        idx_values = [row.pop(index_col) for row in rows]
        # try numeric index
        try:
            idx_values = [float(v) for v in idx_values]
            idx_values = [int(v) if v == int(v) else v for v in idx_values]
        except ValueError:
            pass
        index = Index(idx_values, name=idx_name)
    else:
        index = None

    # Parse columns
    data = {}
    for ci, col_name in enumerate(header):
        raw = [row[ci] for row in rows]
        # try to convert to numeric
        try:
            floats = [float(v) for v in raw]
            # keep as int if all values are integral
            if all(v == int(v) for v in floats):
                data[col_name] = np.array([int(v) for v in floats])
            else:
                data[col_name] = np.array(floats)
        except ValueError:
            data[col_name] = np.array(raw)

    return DataFrame(data, index=index)


def _to_csv(df: DataFrame, filepath: str, index: bool = True) -> None:
    """Write a DataFrame to a CSV file."""

    cols = df._columns.tolist()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if index:
            writer.writerow([""] + cols)
        else:
            writer.writerow(cols)

        for i in range(len(df)):
            row = []
            if index:
                label = df.index[i]
                row.append(label)
            for c in cols:
                val = df._data[c][i]
                val = val.item() if isinstance(val, np.generic) else val
                row.append(val)
            writer.writerow(row)
