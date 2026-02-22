"""LioPandas — a minimal pandas-like library built from scratch using NumPy."""

from .series import Series
from .dataframe import DataFrame
from .io import read_csv
from .merge import merge

__all__ = ["Series", "DataFrame", "read_csv", "merge"]
__version__ = "0.1.0"
