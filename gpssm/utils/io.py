"""Simple IO helpers for saving and loading arrays and objects.

These helpers provide small convenience wrappers used by scripts and tests.
"""

import numpy as np
import os
import pickle
from typing import Any


def save_npz(path: str, **arrays):
    """Save named arrays to a compressed NPZ file.

    Example: `save_npz("out.npz", X=X, y=y)`
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str):
    """Load a compressed NPZ file and return a dict of arrays."""
    return dict(np.load(path))


def save_object(path: str, obj: Any):
    """Save a Python object with pickle."""
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path: str):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
