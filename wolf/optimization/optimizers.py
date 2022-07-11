import inspect
from typing import Callable, Dict, Any

import torch
import torch.nn as nn

from yield_forecasting.utils.logger import logger


def get_default_args(obj: Callable) -> Dict[str, Any]:
    """Returns default args of any callable."""
    signature = inspect.signature(obj)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'RMS': torch.optim.RMSprop,
    'SGD': torch.optim.SGD
}


def get_optimizer(optimizer_name: str, network: nn.Module, **optimizer_kwargs) -> torch.optim.Optimizer:
    """
    Returns the optimizer object.

    Args:
        optimizer_name: the name of optimizer, such as "Adam", "RMS", etc.
        network: the PyTorch network object
        **optimizer_kwargs: the arguments for optimizer, for example "lr=1e-3"

    Returns:
        optimizer object
    """

    try:
        optimizer_cls = OPTIMIZERS[optimizer_name]
    except KeyError:
        raise NotImplementedError(
            f'Given optimizer is not supported. Please select from: {list(OPTIMIZERS.keys())}.'
        )

    optimizer_args = get_default_args(optimizer_cls)
    for k, v in optimizer_kwargs.items():
        if k in optimizer_args:
            optimizer_args[k] = v
        else:
            logger.warning(f'Optimizer {optimizer_name} does not have argument {k}.'
                           f'The value {v} for {k} will be ignored.')

    return optimizer_cls(network.parameters(), **optimizer_args)
