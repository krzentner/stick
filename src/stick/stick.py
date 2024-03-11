import os
import sys
import inspect
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Union, Optional
import warnings
import functools
import datetime
import logging

import stick
from stick.flat_utils import flatten, FlatDict, ScalarTypes

INIT_CALLED = False

# Log levels

LOG_LEVELS = {}

_PY_LOGGER = None


@dataclass(eq=False, order=False, frozen=True)
@functools.total_ordering
class LogLevel:
    name: str
    val: int

    def __eq__(self, other):
        return self.val == other

    def __lt__(self, other):
        return self.val < other

    def __repr__(self):
        return f"LogLevel({self.name}, {self.val})"

    def __hash__(self):
        return hash(self.val)

    def __int__(self):
        return self.val

    def __str__(self):
        return self.name

    @classmethod
    def register(cls, name, val):
        log_level = cls(name, val)
        assert name not in LOG_LEVELS
        LOG_LEVELS[name] = log_level
        logging.addLevelName(val, name)
        return log_level


# Used for extremely noisy logging (maybe still useful for debugging)
TRACE = LogLevel.register("TRACE", 5)
# Used for important results (not an error, but less noisy than INFO)
RESULTS = LogLevel.register("RESULTS", 35)

# These have the same value as the standard library logging levels
DEBUG = LogLevel.register("DEBUG", 10)
INFO = LogLevel.register("INFO", 20)
WARNING = LogLevel.register("WARNING", 30)
ERROR = LogLevel.register("ERROR", 40)
CRITICAL = LogLevel.register("CRITICAL", 50)


def _warn_internal(msg):
    _PY_LOGGER.warn(msg)


def default_run_name():
    main_file = getattr(sys.modules.get("__main__"), "__file__", "interactive")
    file_trail = os.path.splitext(os.path.basename(main_file))[0]
    now = datetime.datetime.now().isoformat()
    return f"{file_trail}_{now}"


def init(log_dir="runs", run_name=None) -> str:
    """Initializes a logger in {log_dir}/{run_name} with default backends.

    Run name will default to the main file and current time in ISO 8601 format.
    """
    if run_name is None:
        run_name = default_run_name()
        if os.path.exists(run_name):
            raise ValueError(
                "Could not create a unique default run name. "
                "Most likely two runs began at the same time."
            )

    run_dir = os.path.abspath(os.path.join(log_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)

    FORMAT = "%(asctime)s %(name)s [%(levelname)-8.8s]: %(message)s"
    global _PY_LOGGER
    _PY_LOGGER = logging.getLogger("stick")
    _PY_LOGGER.setLevel(0)
    formatter = logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=os.path.join(run_dir, "debug.log"))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(int(TRACE))

    _PY_LOGGER.addHandler(file_handler)
    _PY_LOGGER.addHandler(stream_handler)

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(stream_handler)
    warnings_logger.addHandler(file_handler)

    global INIT_CALLED
    if INIT_CALLED:
        _warn_internal(
            "stick.init() already called in this process. Most "
            "likely stick.log() was called before stick.init()."
        )
        return run_dir
    INIT_CALLED = True

    if LOGGER_STACK:
        _warn_internal("logger was already present before stick.init() was called")
    LOGGER_STACK.append(setup_default_logger(log_dir, run_name))
    return run_dir


INIT_EXTRA_CALLED = False


def init_extra(
    log_dir="runs",
    run_name=None,
    config=None,
    wandb_kwargs=None,
    init_dotenv=True,
    init_wandb=True,
    seed_all="if_present",
    create_git_checkpoint=True,
) -> str:
    run_dir = init(log_dir, run_name)
    global INIT_EXTRA_CALLED
    if INIT_EXTRA_CALLED:
        return run_dir
    INIT_EXTRA_CALLED = True

    if config is None:
        config = {}

    if isinstance(config, dict):
        config_as_dict = config
    else:
        config_as_dict = config.__dict__

    if init_dotenv:
        try:
            from dotenv import load_dotenv
        except ImportError:
            _warn_internal("could not import dotenv")
        else:
            load_dotenv()

    if init_wandb:
        try:
            import wandb
        except ImportError:
            _warn_internal("could not import wandb")
            pass
        else:
            if wandb_kwargs is None:
                wandb_kwargs = {}
            wandb_kwargs.setdefault("sync_tensorboard", True)
            wandb_kwargs.setdefault("config", config)
            wandb.init(**wandb_kwargs)

    if create_git_checkpoint:
        try:
            import stick.stick_git

            stick.stick_git.checkpoint_repo(run_dir)
        except ImportError:
            _warn_internal("could not import git, repo was not checkpointed")

    if seed_all:
        assert seed_all is True or seed_all == "if_present"
        MISSING = object()
        seed = MISSING
        try:
            seed = config_as_dict["seed"]
        except KeyError:
            pass
        if seed is not None and seed is not MISSING:
            try:
                seed = int(seed)
            except ValueError:
                warnings.warn(f"Could not convert seed {seed!r} to int")
            else:
                seed_all_imported_modules(seed)
        elif seed_all is True:
            # We were told to seed, and could not
            raise ValueError(
                "Explicitly asked to seed, " "but seed is not present in config"
            )
    log(table="hparams", row=config_as_dict)
    return run_dir


def seed_all_imported_modules(seed: int, make_deterministic: bool = True):
    import random

    random.seed(seed)

    if "numpy" in sys.modules:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.seed(seed)

    if "torch" in sys.modules:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            if make_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    if "tensorflow" in sys.modules:
        try:
            import tensorflow as tf
        except ImportError:
            pass
        else:
            tf.random.set_seed(seed)
            tf.experimental.numpy.random.seed(seed)
            tf.set_random_seed(seed)
            if make_deterministic:
                os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
                os.environ["TF_DETERMINISTIC_OPS"] = "1"

    os.environ["PYTHONHASHSEED"] = str(seed)


LOGGER_STACK = []


def get_logger():
    if "torch" in sys.modules:
        import stick.torch
    if not LOGGER_STACK:
        init()
    return LOGGER_STACK[-1]


def add_output(output_engine):
    get_logger().add_output(output_engine)


class LoggerWarning(UserWarning):
    def __init__(self, msg):
        super().__init__(msg)
        _PY_LOGGER.warn(msg)


class LoggerError(ValueError):
    pass


@dataclass
class Row:
    table_name: str
    raw: Any

    # "Should" be monotonically increasing, but not necessarily sequential
    step: int

    # If someone is manually creating a Row, probably default to INFO (the
    # default logging level)
    log_level: int = INFO

    def as_flat_dict(self, prefix="") -> FlatDict:
        flat_dict = {}
        flatten(self.raw, prefix, flat_dict)
        return flat_dict


class Logger:
    def __init__(self, log_dir, run_name):
        self.log_dir = log_dir
        self.run_name = run_name
        self.fileloc_to_tables: dict[tuple[str, int], str] = {}
        self.tables_to_fileloc: dict[str, tuple[str, int]] = {}
        self.table_to_default_step: defaultdict[str, int] = defaultdict(int)
        self._output_engines: defaultdict[str, list[OutputEngine]] = defaultdict(list)
        self._closed = False

    @property
    def all_output_engines(self):
        return [
            engine for engines in self._output_engines.values() for engine in engines
        ]

    def get_unique_table(self, filename: str, lineno: int) -> str:
        fileloc = (filename, lineno)
        try:
            return self.fileloc_to_tables[fileloc]
        except KeyError:
            pass
        parts = stick.utils.splitall(filename)
        for i in range(1, len(parts)):
            if i == 0:
                table = parts[-1]
            else:
                table = os.path.join(*parts[-i:])
                table = f"{table}:{lineno}"
            if table not in self.tables_to_fileloc:
                self.tables_to_fileloc[table] = fileloc
                self.fileloc_to_tables[fileloc] = table
                return table
        return f"{filename}:{lineno}"

    def get_default_step(self, table: str) -> int:
        step = self.table_to_default_step[table]
        self.table_to_default_step[table] += 1
        return step

    def log_row(self, row):
        for output in self.all_output_engines:
            output.log_row(row)

    def close(self):
        assert not self._closed
        for output in self.all_output_engines:
            output.close()
        self._closed = True

    def __enter__(self):
        LOGGER_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert LOGGER_STACK[-1] is self
        LOGGER_STACK.pop()

    def add_output(self, output):
        self._output_engines[type(output).__name__].append(output)


class OutputEngine:
    def __init__(self, log_level):
        self.log_level = log_level

    def log_row(self, row: Row):
        if row.log_level >= self.log_level:
            self.log_row_inner(row)

    def log_row_inner(self, row: Row):
        pass

    def close(self):
        pass


OUTPUT_ENGINES = {}
LOAD_FILETYPES = {}


def declare_output_engine(output_engine_type):
    """This decorator basically just serves as notice that the output's type
    name is part of its interface."""
    assert issubclass(output_engine_type, OutputEngine)
    OUTPUT_ENGINES[output_engine_type.__name__] = output_engine_type
    return output_engine_type


def setup_default_logger(log_dir, run_name):
    assert INIT_CALLED, "Call init() instead"

    from stick.json_output import JsonOutputEngine
    from stick.pprint_output import PPrintOutputEngine
    from stick.tb_output import TensorBoardOutput

    logger = Logger(log_dir=log_dir, run_name=run_name)
    logger.add_output(JsonOutputEngine(f"{log_dir}/{run_name}/stick.ndjson"))
    try:
        logger.add_output(TensorBoardOutput(log_dir, run_name))
    except ImportError:
        _warn_internal("tensorboard API not installed")
    logger.add_output(PPrintOutputEngine(f"{log_dir}/{run_name}/stick.log"))
    return logger


def load_log_file(
    filename: str, keys: Optional[list[str]] = None
) -> dict[str, list[ScalarTypes]]:
    # Import the json output engine, since it has no external deps
    import stick.json_output

    _, ext = os.path.splitext(filename)
    if ext in LOAD_FILETYPES:
        return LOAD_FILETYPES[ext](filename, keys)
    else:
        raise ValueError(
            f"Unknown filetype {ext}. Perhaps you need to load "
            "a stick backend or use one of the well known log "
            "types (.ndjson)"
        )


def log(
    table: Optional[Union[str, dict, Row]] = None,
    row: Optional[Union[dict, Row]] = None,
    step: Optional[int] = None,
    level: Optional[Union[int, LogLevel]] = None,
):
    """Primary logging entrypoint.

    Shorthand for constructing a row and calling get_logger().log_row(row).
    Will deduce any non-provided arguments.

    If a Row object is provided, it contains all required fields and will be
    passed to log_row, with arguments overriding the row fields if provided.

    If the first argument is a str, it becomes the table name.
    If the first argument is a dict, the table name is derived from the calling
    file and line number.

    If the row is not provided, locals from the calling function will be
    logged, and the log_level will be set to TRACE (log level 5).
    In other cases, level defaults to INFO (log level 10), which is also the
    default value for most output engines.

    Step determines the "X axis" of the logged data. If not provided it will
    automatically be incremented every call with the same table.

    Examples:

    # Will log all local variables at log level TRACE
    log()

    # Will log all local variables at log level TRACE to table called 'grad_step'
    log('grad_step')

    # Will log into table based on current file and line number at log level INFO
    log({'x': x})
    """

    logger = get_logger()

    # Most of this function is magic to figure out how to produce a Row object
    # based on minimal arguments.

    # Special case only a row-like passed in as the first argument
    # e.g. log({'x': x}) or log(Row('grad_step', {'x': x}))
    if isinstance(table, (dict, Row)) and row is None and step is None:
        row = table
        table = None

    # Check all the types
    if table is not None:
        assert isinstance(table, str)
    if row is not None and not isinstance(row, (dict, Row)):
        raise ValueError(
            f"Unsupported row type {type(row)}. "
            "Use a dictionary or inherit from stick.Row."
        )
    if step is not None:
        assert isinstance(step, int)

    # Check what we have
    if isinstance(row, Row):
        if level is not None:
            row = replace(row, log_level=level)
        # We have a row, so we won't need to guess any info
        # Arguments might still override parts of the Row object
        # e.g. log(Row('grad_step', {'x': x}), table='grad_step2')
        # e.g. log(Row('grad_step', {'x': x}), step=0)
        if table is None and step is None:
            # Case 1: We have a row, just log it
            logger.log_row(row=row)
        elif isinstance(table, str) and step is None:
            # Case 2: User provided the table and not a step, recompute the step
            step = logger.get_default_step(table)
            logger.log_row(replace(row, table_name=table, step=step))
        elif isinstance(table, str) and isinstance(step, int):
            # Case 3: User provided the table and step
            logger.log_row(replace(row, table_name=table, step=step))
    elif isinstance(table, str) and isinstance(row, dict):
        # No inspecting required, just need to figure out the step
        # e.g. log('grad_step', {'x': x})
        if step is None:
            step = logger.get_default_step(table)
        if level is None:
            level = INFO
        logger.log_row(Row(raw=row, table_name=table, step=step, log_level=level))
    else:
        # We do not have a Row, row is either a dict or None
        # We need to figure out either our table name or row contents using
        # inspect.
        # e.g. log({'x': x}) or log() or log(table='x')
        stack = inspect.stack()
        frame_info = stack[1]
        if table is None:
            table = logger.get_unique_table(frame_info.filename, frame_info.lineno)
        if step is None:
            step = logger.get_default_step(table)
        if row is None:
            row = frame_info.frame.f_locals
            if level is None:
                level = TRACE
            # Since we're inferring the row from the locals, there's likely a
            # `self` variable. Replace the `self` key with the type name of the
            # self, since that's more informative.
            if "self" in row:
                type_name = type(row["self"]).__name__
                if type_name not in row:
                    row[type_name] = row["self"]
                    del row["self"]
        if level is None:
            level = INFO
        logger.log_row(Row(raw=row, table_name=table, step=step, log_level=level))


def test_log_pprint(tmp_path):
    from stick.pprint_output import PPrintOutputEngine
    import io

    hi = "HI ^_^"
    f = io.StringIO()
    with Logger(log_dir=tmp_path, run_name="test_log_pprint") as logger:
        logger.add_output(PPrintOutputEngine(f))
        log()
        content1 = f.getvalue()
        assert hi in content1
        log()
        content2 = f.getvalue()
        assert hi in content2
        assert content2.startswith(content1)
        assert len(content2) > len(content1)
        print(content2)


def test_log_json(tmp_path):
    from stick.json_output import JsonOutputEngine
    import io

    hi = "HI ^_^"
    f = io.StringIO()
    with Logger(log_dir=tmp_path, run_name="test_log_json") as logger:
        logger.add_output(JsonOutputEngine(f))
        log()
        content1 = f.getvalue()
        assert hi in content1
        log()
        content2 = f.getvalue()
        assert hi in content2
        assert content2.startswith(content1)
        assert len(content2) > len(content1)
        print(content2)
