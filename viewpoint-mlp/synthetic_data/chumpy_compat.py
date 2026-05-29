# Helper file to handle legacy chumpy-based model files without requiring chumpy as a dependency
import pickle

import numpy as np
import torch


class DummyChumpy(object):
    """Fallback object used when unpickling legacy chumpy-based model files."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


class ChumpyBypasser(pickle.Unpickler):
    """Unpickler that swaps chumpy classes with DummyChumpy."""

    def find_class(self, module, name):
        if module == "chumpy" or module.startswith("chumpy."):
            return DummyChumpy
        return super().find_class(module, name)


def force_to_tensor(obj):
    """Recursively finds numeric payloads in nested legacy chumpy containers."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    if isinstance(obj, (float, int)):
        return torch.tensor([obj]).float()

    if isinstance(obj, DummyChumpy):
        for attr in ["r", "_obj", "v", "dat"]:
            if hasattr(obj, attr):
                return force_to_tensor(getattr(obj, attr))
        for value in obj.__dict__.values():
            if isinstance(value, (np.ndarray, DummyChumpy)):
                return force_to_tensor(value)

    return torch.from_numpy(np.array(obj)).float()
