from typing import Dict
from pytorch_warmup import UntunedLinearWarmup, UntunedExponentialWarmup, RAdamWarmup, LinearWarmup, BaseWarmup

import torch
from torch.optim.lr_scheduler import (
    _LRScheduler,
    LambdaLR,
    MultiStepLR,
    StepLR,
    MultiplicativeLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,

)

LR_SCHEDULERS = {
    'LambdaLR': LambdaLR,  # after epoch
    'MultiplicativeLR': MultiplicativeLR,  # after epoch
    'StepLR': StepLR,  # after epoch
    'MultiStepLR': MultiStepLR,  # after epoch
    'ReduceLROnPlateau': ReduceLROnPlateau,  # after epoch
    'CyclicLR': CyclicLR,  # after batch
    'OneCycleLR': OneCycleLR,  # after batch
    'CosineAnnealingLR': CosineAnnealingLR,  # after batch
    'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts,  # after batch
}

WARMUP_SCHEDULERS = {
    "UnTunedLinear": UntunedLinearWarmup,
    "UnTunedExponential": UntunedExponentialWarmup,
    "RAdam": RAdamWarmup,
    "Linear": LinearWarmup
}


EPOCH_UPDATE_SCHEDULERS = frozenset(('LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ReduceLROnPlateau'))
METRIC_BASED_SCHEDULERS = frozenset(('ReduceLROnPlateau', ))


def get_lr_scheduler(lr_scheduler_name: str,
                     optimizer: torch.optim.Optimizer,
                     **lr_scheduler_kwargs: Dict) -> _LRScheduler:
    """Returns the LR_Scheduler class based on its name."""
    try:
        return LR_SCHEDULERS[lr_scheduler_name](optimizer=optimizer, **lr_scheduler_kwargs)
    except KeyError:
        raise ValueError(
            f'Given LR scheduler {lr_scheduler_name} is not supported.'
            f'Please select from: {" | ".join(LR_SCHEDULERS.keys())}.'
        )


def get_warmup_scheduler(warmup_name: str, optimizer: torch.optim.Optimizer, **warmup_kwargs: Dict) -> BaseWarmup:
    try:
        return WARMUP_SCHEDULERS[warmup_name](optimizer=optimizer, **warmup_kwargs)
    except KeyError:
        raise ValueError(
            f'Given WarmUp scheduler {warmup_name} is not supported.'
            f'Please select from: {" | ".join(WARMUP_SCHEDULERS.keys())}.'
        )
