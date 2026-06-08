import os
import tempfile
import numpy as np
from gpssm.utils.io import save_npz, load_npz, save_object, load_object


def test_save_and_load_npz_roundtrip():
    X = np.arange(6).reshape(3, 2)
    y = np.array([1.0, 2.0, 3.0])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.npz")
        save_npz(path, X=X, y=y)
        data = load_npz(path)
        assert "X" in data and "y" in data
        assert np.array_equal(data["X"], X)
        assert np.array_equal(data["y"], y)


def test_save_and_load_object_roundtrip():
    obj = {"a": 1, "b": [1, 2, 3]}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "obj.pkl")
        save_object(path, obj)
        loaded = load_object(path)
        assert loaded == obj
