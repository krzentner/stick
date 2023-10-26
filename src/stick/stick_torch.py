from stick import declare_processor, collapse_prefix
import torch
from torch import nn

LOG_HISTOGRAMS = set([collapse_prefix(prefix=['global'],
                                      key='histogram')])
LOG_IMAGES = set([collapse_prefix(prefix=['global'],
                                  key='image')])


@declare_processor(torch.Tensor)
def process_tensor(logger, prefix, key, tensor):
    logger.logkv('mean', torch.mean(tensor).item(), [key])
    logger.logkv('std', torch.std(tensor).item(), [key])
    full_key = collapse_prefix(prefix=prefix, key=key)
    @logger.map_output('DowelOutputEngine')
    def log_dowel(output):
        import dowel
        if full_key in LOG_HISTOGRAMS:
            output.log(prefix, key, value,
                        dowel.Histogram(tensor.detach().numpy()))
    @logger.map_output('TensorBoardXOutput')
    def log_tbx(output):
        if full_key in LOG_HISTOGRAMS:
            output.writer.add_histogram(full_key, tensor.detach().numpy(),
                                        output.step)
        if full_key in LOG_IMAGES:
            output.writer.add_image(full_key, tensor.detach().numpy(),
                                    output.step)


@declare_processor(nn.Module)
def process_module(logger, prefix, key, module):
    for (name, param) in module.named_parameters():
        logger.logkv(name, param, [key])
        if param.grad is not None:
            logger.logkv('grad', param.grad, [key, name])


@declare_processor(torch.optim.Optimizer)
def process_optimizer(logger, prefix, key, optimizer):
    state_dict = optimizer.state_dict()
    state = state_dict['state']
    param_groups = state_dict['param_groups']
    for (k, v) in state.items():
        logger.logkv(k, v, [key])
    for (i, group) in enumerate(param_groups):
        for (k, v) in group.items():
            logger.logkv(k, v, [key, i])
