from stick import OutputEngine, declare_output_engine, INFO
from stick.stick import _warn_internal

SummaryWriter = None

DEFAULT_LOG_LEVEL = INFO


@declare_output_engine
class TensorBoardOutput(OutputEngine):
    def __init__(self, log_dir, run_name, flush_secs=120, log_level=None):
        # Allow default to be overridden
        # TODO: Find a better API for this.
        if log_level is None:
            log_level = DEFAULT_LOG_LEVEL
        super().__init__(log_level=log_level)
        global SummaryWriter
        try:
            if SummaryWriter is None:
                # Apparently this is necessary?
                import caffe2
                from torch.utils.tensorboard import SummaryWriter
        except (ImportError, ModuleNotFoundError):
            pass
        try:
            if SummaryWriter is None:
                from tensorboardX import SummaryWriter
        except ImportError:
            pass
        try:
            if SummaryWriter is None:
                from tf.summary import SummaryWriter
        except ImportError:
            pass

        if SummaryWriter is None:
            raise ImportError("Could not find tensorboard API")

        self.writer = SummaryWriter(f"{log_dir}/{run_name}", flush_secs=flush_secs)
        self.run_name = run_name
        print(f"TensorBoardOutput [{self.log_level}] in: " f"{log_dir}/{run_name}")

    def log_row_inner(self, row):
        if row.table_name == "hparams":
            flat_dict = row.as_flat_dict()
            hparams = {
                k: v for (k, v) in flat_dict.items() if not k.startswith("metric")
            }
            metrics = {k: v for (k, v) in flat_dict.items() if k.startswith("metric")}
            # Handle tensorboardX API difference
            try:
                self.writer.add_hparams(hparams, metrics, run_name="hparams")
            except TypeError:
                self.writer.add_hparams(hparams, metrics, name="hparams")
        else:
            for k, v in row.as_flat_dict().items():
                if v is not None and not isinstance(v, str):
                    try:
                        self.writer.add_scalar(f"{row.table_name}/{k}", v, row.step)
                    except (TypeError, NotImplementedError) as ex:
                        _warn_internal(f"Could not log key {k} in TensorBoard: {ex}")
        self.writer.flush()

    def close(self):
        """Flush all the events to disk and close the file."""
        self.writer.close()
