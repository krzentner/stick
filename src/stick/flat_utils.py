"""Handles converting raw data to flattened dictionaries for logging."""
import sys
import fnmatch
from typing import Any, Union

from stick.utils import warn_internal

STICK_PREPROCESSOR = "stick_preprocess"

SKIP = object()

PROCESSORS = {}

MAX_SEQ_LEN = 8


def declare_processor(type_description: Union[str, type], monkey_patch: bool = True):
    if isinstance(type_description, str):
        type_str = type_description
    else:
        type_str = type_string(type_description)

    def decorator(processor):
        assert type_str not in PROCESSORS
        PROCESSORS[type_str] = processor

        if monkey_patch:
            # Try to monkey-patch STICK_PREPROCESSOR method onto type
            parts = type_str.split(".")
            try:
                obj = sys.modules[parts[0]]
                for p in parts[1:]:
                    obj = getattr(obj, p, None)
                setattr(obj, STICK_PREPROCESSOR, processor)
            except (KeyError, AttributeError, TypeError) as ex:
                warn_internal(
                    f"Coudld not money-patch processor to type {type_str!r}: {ex}"
                )

        return processor

    return decorator


def type_string(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


def is_instance_str(obj, type_names):
    """An isinstance check that does not require importing the type's module."""
    obj_type_str = type_string(obj)
    if isinstance(type_names, str):
        return fnmatch.fnmatch(obj_type_str, type_names)
    else:
        for type_name in type_names:
            if fnmatch.fnmatch(obj_type_str, type_name):
                return True
        return False


# Keep these this list and type synchronized
ScalarTypes = (type(None), str, float, int, bool)
FlatDict = dict[str, Union[None, str, float, int, bool]]


def flatten(src: Any, prefix: str, dst: FlatDict):
    """Lossfully flatten a dictionary."""
    if prefix == "_":
        return
    if isinstance(src, ScalarTypes):
        key = prefix
        i = 1
        while key in dst:
            key = f"{prefix}_{i}"
            i += 1
        dst[key] = src
    elif isinstance(src, dict):
        for k, v in src.items():
            if isinstance(k, int):
                continue
            # Since we often log locals, there's likely a `self` variable.
            # Replace the `self` key with the type name of the self, since
            # that's more informative.
            if k == "self":
                k = type(v).__name__
            try:
                if prefix:
                    flat_k = f"{prefix}/{k}"
                else:
                    flat_k = k
            except ValueError:
                pass
            else:
                flatten(v, flat_k, dst)
    elif isinstance(src, (list, tuple)):
        for i, v in enumerate(src):
            flat_k = f"{prefix}[{i}]"
            flatten(v, prefix, dst)
    else:
        processor = PROCESSORS.get(type_string(src), None)
        if processor is not None:
            processor(src, prefix, dst)
        else:
            processor = getattr(src, STICK_PREPROCESSOR, None)
            if processor is not None:
                processor(prefix, dst)
