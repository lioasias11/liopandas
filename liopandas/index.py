"""Index — labelled axis for Series and DataFrame."""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Sequence, Union


class Index:
    """An immutable sequence of labels used as rows/columns axis."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, data: Optional[Sequence] = None, name: Optional[str] = None):
        if data is None:
            data = []
        self._labels = np.array(list(data))
        self.name = name
        # label -> first position  (rebuilt lazily)
        self._label_map: Optional[dict] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_map(self) -> dict:
        if self._label_map is None:
            self._label_map = {}
            for i, lab in enumerate(self._labels):
                key = lab.item() if isinstance(lab, np.generic) else lab
                if key not in self._label_map:
                    self._label_map[key] = i
        return self._label_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_loc(self, label) -> int:
        """Return the integer position for *label*."""
        m = self._build_map()
        if label not in m:
            raise KeyError(label)
        return m[label]

    def get_locs(self, labels: Sequence) -> List[int]:
        """Return a list of integer positions for each label."""
        return [self.get_loc(l) for l in labels]

    def __contains__(self, item) -> bool:
        return item in self._build_map()

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self):
        for lab in self._labels:
            yield lab.item() if isinstance(lab, np.generic) else lab

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            val = self._labels[key]
            return val.item() if isinstance(val, np.generic) else val
        if isinstance(key, slice):
            return Index(self._labels[key], name=self.name)
        if isinstance(key, (list, np.ndarray)):
            return Index(self._labels[key], name=self.name)
        raise TypeError(f"Invalid key type: {type(key)}")

    def __repr__(self) -> str:
        items = list(self)
        return f"Index({items})"

    def __eq__(self, other):
        if isinstance(other, Index):
            return np.array_equal(self._labels, other._labels)
        return NotImplemented

    def tolist(self) -> list:
        return [x.item() if isinstance(x, np.generic) else x for x in self._labels]

    def copy(self) -> "Index":
        return Index(self._labels.copy(), name=self.name)

    @property
    def values(self) -> np.ndarray:
        return self._labels

    def append(self, other: "Index") -> "Index":
        """Return a new Index by concatenating *self* and *other*."""
        return Index(np.concatenate([self._labels, other._labels]), name=self.name)

    def reset(self, length: Optional[int] = None) -> "Index":
        """Return a default RangeIndex 0..length-1."""
        n = length if length is not None else len(self)
        return Index(range(n))
