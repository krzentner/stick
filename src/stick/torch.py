"""Enhances stick.flatten to handle common torch types."""
from stick.flat_utils import flatten, declare_processor
from stick.utils import warn_internal

try:
    import torch
except ImportError as ex:
    warn_internal("stick.torch imported, but torch could not be imported")
    warn_internal(ex)


@declare_processor("torch.Tensor")
def process_tensor(tensor, key, dst):
    if tensor.flatten().shape == (1,):
        dst[key] = tensor.flatten().item()
    else:
        dst[f"{key}.mean"] = tensor.float().mean().item()
        try:
            dst[f"{key}.min"] = tensor.min().item()
            dst[f"{key}.max"] = tensor.max().item()
        except RuntimeError:
            pass
        try:
            dst[f"{key}.std"] = tensor.float().std().item()
        except RuntimeError:
            pass


@declare_processor("torch.nn.Module")
def process_module(module, key, dst):
    for name, param in module.named_parameters():
        flatten(param, f"{key}.{name}", dst)
        if param.grad is not None:
            flatten(param.grad, f"{key}.{name}.grad", dst)


@declare_processor("torch.optim.Optimizer")
def process_optimizer(optimizer, key, dst):
    flatten(optimizer.state_dict(), key, dst)
