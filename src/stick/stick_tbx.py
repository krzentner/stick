import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import tensorboardX as tbX

from stick import OutputEngine, declare_output_engine, collapse_prefix, LoggerWarning

DEFAULT_X_AXIS_VALUE = -1

@declare_output_engine
class TensorBoardXOutput(OutputEngine):

    def __init__(self,
                 log_dir,
                 x_axis=None,
                 additional_x_axes=None,
                 flush_secs=120,
                 histogram_samples=1e3):
        if x_axis is None:
            assert not additional_x_axes, (
                'You have to specify an x_axis if you want additional axes.')

        additional_x_axes = additional_x_axes or []

        self.writer = tbX.SummaryWriter(log_dir, flush_secs=flush_secs)
        self._x_axis = x_axis
        self._additional_x_axes = additional_x_axes
        self._default_step = 0
        self._histogram_samples = int(histogram_samples)

        self._warned_once = set()
        self._disable_warnings = False
        self._x_axes_values = {
            x: DEFAULT_X_AXIS_VALUE for x in self._additional_x_axes
        }
        self._x_axes_values[self._x_axis] = DEFAULT_X_AXIS_VALUE

    @property
    def step(self):
        return self._x_axes_values[self._x_axis]

    def log(self, prefix, key, value):
        full_key = collapse_prefix(prefix=prefix, key=key)
        if full_key in self._x_axes_values:
            self._x_axes_values[full_key] = value
        if full_key != self._x_axis:
            for (k, x) in self._x_axes_values.items():
                if k == self._x_axis:
                    self._record_kv(full_key, value, x)
                else:
                    self._record_kv(f'{full_key}/{x}', value, x)
        else:
            self.dump()

    def _record_kv(self, key, value, step):
        if isinstance(value, np.ScalarType):
            self.writer.add_scalar(key, value, step)
        elif isinstance(value, plt.Figure):
            self.writer.add_figure(key, value, step)
        elif isinstance(value, scipy.stats._distn_infrastructure.rv_frozen):
            shape = (self._histogram_samples, ) + value.mean().shape
            self.writer.add_histogram(key, value.rvs(shape), step)
        elif isinstance(value, scipy.stats._multivariate.multi_rv_frozen):
            self.writer.add_histogram(key, value.rvs(self._histogram_samples),
                                       step)

    def log_tf_graph(self, graph):
        graph_def = graph.as_graph_def(add_shapes=True)
        event = tbX.proto.event_pb2.Event(
            graph_def=graph_def.SerializeToString())
        self.writer.file_writer.add_event(event)

    def dump(self, step=None):
        # Flush output files
        for w in self.writer.all_writers.values():
            w.flush()

        self._default_step += 1
        if self._default_step > 1:
            self._check_for_updates()

    def _check_for_updates(self):
        for (k, v) in self._x_axes_values:
            if v == DEFAULT_X_AXIS_VALUE:
                self._warn(f'X-Axis {k} was requested, '
                            'but has not been updated')

    def close(self):
        """Flush all the events to disk and close the file."""
        self.writer.close()
        self._check_for_updates()

    def _warn(self, msg, stacklevel=1):
        """Warns the user using warnings.warn."""
        if not self._disable_warnings and msg not in self._warned_once:
            warnings.warn(colorize(msg, 'yellow'),
                          NonexistentAxesWarning,
                          stacklevel=stacklevel)
        self._warned_once.add(msg)
        return msg


class NonexistentAxesWarning(LoggerWarning):
    """Raise when the specified x axes do not exist in the tabular."""
