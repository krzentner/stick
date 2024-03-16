"""Enhances stick.flatten to handle numpy arrays."""
from stick.flat_utils import flatten, declare_processor
from stick.utils import warn_internal

try:
    import numpy as np
except ImportError as ex:
    warn_internal("stick.np imported, but numpy could not be imported")
    warn_internal(ex)


@declare_processor("numpy.ndarray")
def process_tensor(array, key, dst):
    if array.flatten().shape == (1,):
        dst[key] = array.flatten()[0]
    else:
        dst[f"{key}.mean"] = array.astype("float").mean()
        try:
            dst[f"{key}.min"] = array.min()
            dst[f"{key}.max"] = array.max()
        except RuntimeError:
            pass
        try:
            dst[f"{key}.std"] = array.astype("float").std()
        except RuntimeError:
            pass
