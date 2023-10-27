from stick import OutputEngine, declare_output_engine
SummaryWriter = None

@declare_output_engine
class TensorBoardOutput(OutputEngine):

    def __init__(self,
                 log_dir,
                 flush_secs=120,
                 histogram_samples=1e2):
        global SummaryWriter
        try:
            if SummaryWriter is None:
                from torch.utils.tensorboard import SummaryWriter
        except ImportError:
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

        self.writer = SummaryWriter(log_dir, flush_secs=flush_secs)
        self._histogram_samples = int(histogram_samples)

    def log_row(self, row):
        for k, v in row.as_flat_dict().items():
            self.writer.add_scalar(k, v, row.step)
        self.writer.flush()

    def close(self):
        """Flush all the events to disk and close the file."""
        self.writer.close()
