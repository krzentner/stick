import enum
import json
import time


from stick import OutputEngine, declare_output_engine

@declare_output_engine
class JsonOutputEngine(OutputEngine):

    def __init__(self, filename):
        self._filename = filename
        self._file = open(filename, 'a+')

    def log(self, prefix, key, value):
        msg = {
            'prefix': prefix,
            'key': key,
            'value': value,
            'localtime': time.localtime()
        }
        json.dump(msg, fp=self._file, cls=LogEncoder)
        self._file.write('\n')

    def close(self):
        self._file.close()


class LogEncoder(json.JSONEncoder):
    """Encoder to be used as cls in json.dump.

    Args:
        args (object): Passed to super class.
        kwargs (dict): Passed to super class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._markers = {}

    # Modules whose contents cannot be meaningfully or safelly jsonified.
    BLOCKED_MODULES = {
        'tensorflow',
        'ray',
        'itertools',
        'weakref',
    }

    def default(self, o):
        """Perform JSON encoding.

        Args:
            o (object): Object to encode.

        Raises:
            TypeError: If `o` cannot be turned into JSON even using `repr(o)`.

        Returns:
            dict or str or float or bool: Object encoded in JSON.

        """
        # Why is this method hidden? What does that mean?
        # pylint: disable=method-hidden
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-return-statements
        # This circular reference checking code was copied from the standard
        # library json implementation, but it outputs a repr'd string instead
        # of ValueError on a circular reference.
        if isinstance(o, (int, bool, float, str)):
            return o
        else:
            markerid = id(o)
            if markerid in self._markers:
                return 'circular ' + repr(o)
            else:
                self._markers[markerid] = o
                try:
                    return self._default_inner(o)
                finally:
                    del self._markers[markerid]

    def _default_inner(self, o):
        """Perform JSON encoding.

        Args:
            o (object): Object to encode.

        Raises:
            TypeError: If `o` cannot be turned into JSON even using `repr(o)`.
            ValueError: If raised by calling repr on an object.

        Returns:
            dict or str or float or bool: Object encoded in JSON.

        """
        # Why is this method hidden? What does that mean?
        # pylint: disable=method-hidden
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-return-statements
        # This circular reference checking code was copied from the standard
        # library json implementation, but it outputs a repr'd string instead
        # of ValueError on a circular reference.
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError as err:
            if isinstance(o, dict):
                data = {}
                for (k, v) in o.items():
                    if isinstance(k, str):
                        data[k] = self.default(v)
                    else:
                        data[repr(k)] = self.default(v)
                return data
            elif type(o).__module__.split('.')[0] in self.BLOCKED_MODULES:
                return repr(o)
            elif isinstance(o, type):
                return {'$typename': o.__module__ + '.' + o.__name__}
            elif type(o).__module__ == 'numpy' and type(o).__name__.startswith('float'):
                # For some reason these aren't natively considered
                # serializable.
                # JSON doesn't actually have ints, so always use a float.
                return float(o)
            elif type(o).__module__ == 'numpy' and 'int' in type(o).__name__:
                return int(o)
            elif type(o).__module__ == 'numpy' and 'bool' in type(o).__name__:
                return bool(o)
            elif isinstance(o, enum.Enum):
                return {
                    '$enum':
                    o.__module__ + '.' + o.__class__.__name__ + '.' + o.name
                }
            elif type(o).__module__ == 'numpy':
                # Probably an array
                return repr(o)
            elif hasattr(o, '__dict__') or hasattr(o, '__slots__'):
                obj_dict = getattr(o, '__dict__', None)
                if obj_dict is not None:
                    data = {k: self.default(v) for (k, v) in obj_dict.items()}
                else:
                    data = {
                        s: self.default(getattr(o, s))
                        for s in o.__slots__
                    }
                t = type(o)
                data['$type'] = t.__module__ + '.' + t.__name__
                return data
            elif callable(o) and hasattr(o, '__name__'):
                if getattr(o, '__module__', None) is not None:
                    return {'$function': o.__module__ + '.' + o.__name__}
                else:
                    return repr(o)
            else:
                try:
                    # This case handles many built-in datatypes like deques
                    return [self.default(v) for v in list(o)]
                except TypeError:
                    pass
                try:
                    # This case handles most other weird objects.
                    return repr(o)
                except TypeError:
                    pass
                raise err
