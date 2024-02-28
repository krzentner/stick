"""Handles converting raw data to flattened dictionaries for logging."""
from typing import Any, Union

STICK_PREPROCESSOR = "stick_preprocess"

SKIP = object()

PROCESSORS = {}

MAX_SEQ_LEN = 8


def declare_processor(type_to_process):
    def decorator(processor):
        assert type_to_process not in PROCESSORS
        PROCESSORS[type_to_process] = processor
        setattr(type_to_process, STICK_PREPROCESSOR, processor)
        return processor

    return decorator


# Keep these this list and type synchronized
ScalarTypes = (type(None), str, float, int, bool)
FlatDict = dict[str, Union[None, str, float, int, bool]]


def flatten(src: Any, prefix: str, dst: FlatDict):
    """Lossfully flatten a dictionary."""
    if prefix == '_':
        return
    if isinstance(src, ScalarTypes):
        key = prefix
        i = 1
        while key in dst:
            key = f"{prefix}_{i}"
            i += 1
        dst[key] = src
    elif isinstance(src, dict) and (len(src) <= MAX_SEQ_LEN or not prefix):
        for k, v in src.items():
            if isinstance(k, int):
                continue
            try:
                if prefix:
                    flat_k = f"{prefix}/{k}"
                else:
                    flat_k = k
            except ValueError:
                pass
            else:
                flatten(v, flat_k, dst)
    elif isinstance(src, (list, tuple)) and len(src) <= MAX_SEQ_LEN:
        for i, v in enumerate(src):
            flat_k = f"{prefix}[{i}]"
            flatten(v, prefix, dst)
    else:
        processor = PROCESSORS.get(type(src), None)
        if processor is not None:
            processor(src, prefix, dst)
        else:
            processor = getattr(src, STICK_PREPROCESSOR, None)
            if processor is not None:
                processor(prefix, dst)
