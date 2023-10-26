import sys

class LoggerWarning(UserWarning):
    pass

class LoggerError(ValueError):
    pass


class Logger:

    def __init__(self, outputs=None, prefix=()):
        self.outputs = outputs or {}
        prefix = list(prefix)
        self._prefix = prefix
        self._prev_global_logger = None
        self._closed = False

    @property
    def prefix(self):
        return self._prefix

    def logkv(self, key, value, add_prefix=None):
        if add_prefix is None:
            prefix = self._prefix
        else:
            prefix = self._prefix + add_prefix
        processor = None
        try:
            processor = PROCESSORS[type(value)]
            # Note that processor could be explicitly set to None here, to skip
            # the below loop for common types
        except KeyError:
            for (target_type, p) in PROCESSORS.items():
                if isinstance(value, target_type):
                    processor = p
        if processor is None:
            for output in self.outputs.values():
                output.log(prefix, key, value)
        else:
            processor(self, prefix, key, value)

    def close(self):
        assert not self._closed
        for output in self.outputs.values():
            output.close()
        self._closed = True

    def with_subprefix(self, subprefix):
        assert isinstance(subprefix, list)
        return Logger(
            outputs=self.outputs,
            prefix=self._prefix + subprefix)

    def __enter__(self):
        assert self._prev_global_logger is None
        global GLOBAL_LOGGER
        self._prev_global_logger = GLOBAL_LOGGER
        GLOBAL_LOGGER = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global GLOBAL_LOGGER
        GLOBAL_LOGGER = self._prev_global_logger
        self._prev_global_logger = None

    def add_output(self, output):
        self.outputs[type(output).__name__] = output

    def map_output(self, output_name):
        def decorator(callback):
            try:
                output = self.outputs[output_name]
            except KeyError:
                return None
            else:
                return callback(output)


PROCESSORS = {float: None, int: None, str: None}


def declare_processor(type_to_process):
    def decorator(processor):
        assert type_to_process not in PROCESSORS
        PROCESSORS[type_to_process] = processor
        return processor


class OutputEngine:

    def log(self, prefix, key, value):
        pass

    def close(self):
        pass


OUTPUT_ENGINES = {}

def declare_output_engine(output_engine_type):
    """This decorator basically just serves as notice that the output's type
    name is part of its interface."""
    assert issubclass(output_engine_type, OutputEngine)
    OUTPUT_ENGINES[output_engine_type.__name__] = output_engine_type
    return output_engine_type


GLOBAL_LOGGER = None


def get_logger():
    assert GLOBAL_LOGGER is not None
    return GLOBAL_LOGGER


def set_logger(logger):
    global GLOBAL_LOGGER
    GLOBAL_LOGGER = logger

def prefix(p):
    return get_logger().with_subprefix([p])

def logkv(key, value):
    return get_logger().logkv(key, value)

def setup_logger(directory='.'):
    from stick_json import JsonOutputEngine
    logger = Logger()
    logger.add_output(JsonOutputEngine(f'{directory}/stick.ndjson'))
    if 'dowel' in sys.modules:
        from stick_dowel import global_dowel_output
        logger.add_output(global_dowel_output())
    set_logger(logger)
    return logger

def collapse_prefix(*, prefix, key):
    return '/'.join(prefix + [key])

def test_log():
    from stick_json import JsonOutputEngine
    import json
    import tempfile
    d = tempfile.mkdtemp()
    logger = Logger()
    logger.add_output(JsonOutputEngine(f'{d}/test.ndjson'))
    logger.logkv('hello', 'world')
    logger.close()
    contents = []
    with open(f'{d}/test.ndjson') as f:
        for line in f:
            contents.append(json.loads(line))
    assert len(contents) == 1
    assert contents[0]['key'] == 'hello'
    assert contents[0]['value'] == 'world'
