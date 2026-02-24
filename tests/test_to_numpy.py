import numpy as np
import pytest
from liopandas import DataFrame, Series

class TestSeriesToNumpy:
    def test_basic(self):
        s = Series([1, 2, 3])
        arr = s.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([1, 2, 3]))

    def test_dtype(self):
        s = Series([1, 2, 3])
        arr = s.to_numpy(dtype='float32')
        assert arr.dtype == np.float32
        assert np.array_equal(arr, np.array([1, 2, 3], dtype='float32'))

    def test_copy(self):
        s = Series([1, 2, 3])
        arr = s.to_numpy(copy=True)
        arr[0] = 99
        assert s.iloc[0] == 1
        
        arr_view = s.to_numpy(copy=False)
        arr_view[0] = 100
        assert s.iloc[0] == 100

    def test_na_value(self):
        s = Series([1.0, np.nan, 3.0])
        arr = s.to_numpy(na_value=-1)
        assert np.array_equal(arr, np.array([1.0, -1.0, 3.0]))
        
    def test_na_value_object(self):
        s = Series([1, None, 3], dtype=object)
        arr = s.to_numpy(na_value="missing")
        assert np.array_equal(arr, np.array([1, "missing", 3], dtype=object))

class TestDataFrameToNumpy:
    def test_basic(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        arr = df.to_numpy()
        assert arr.shape == (2, 2)
        assert np.array_equal(arr, np.array([[1, 3], [2, 4]]))

    def test_dtype(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        arr = df.to_numpy(dtype='float64')
        assert arr.dtype == np.float64
        assert np.array_equal(arr, np.array([[1, 3], [2, 4]], dtype='float64'))

    def test_na_value(self):
        df = DataFrame({"a": [1.0, np.nan], "b": [np.nan, 4.0]})
        arr = df.to_numpy(na_value=0)
        expected = np.array([[1.0, 0.0], [0.0, 4.0]])
        assert np.array_equal(arr, expected)

    def test_mixed_types(self):
        df = DataFrame({"a": [1, 2], "b": ["x", "y"]})
        arr = df.to_numpy()
        assert arr.dtype == object
        assert arr[0, 1] == "x"

    def test_na_value_mixed(self):
        df = DataFrame({"a": [1, None], "b": ["x", np.nan]})
        arr = df.to_numpy(na_value="MISSING")
        assert arr[0, 1] == "x"
        assert arr[1, 0] == "MISSING"
        assert arr[1, 1] == "MISSING"

    def test_copy(self):
        df = DataFrame({"a": [1, 2]})
        arr = df.to_numpy(copy=True)
        arr[0, 0] = 99
        assert df.iloc[0, 0] == 1
