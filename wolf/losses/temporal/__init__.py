from typing import Dict

import torch

from .base import TemporalLoss
from .tile_level import TILE_LEVEL_LOSSES
from ...constants import TaskTypes


def get_temporal_loss(task: TaskTypes, loss_name: str, **loss_kwargs: Dict) -> TemporalLoss:
    """Returns the loss class based on its name."""
    loss_kwargs = {
        k: (torch.tensor(v, device='cpu') if isinstance(v, list) else v)
        for k, v in loss_kwargs.items()
    }

    if task in TaskTypes.get_tile_level_tasks():
        loss_pool = TILE_LEVEL_LOSSES[task]
    else:
        raise NotImplementedError(f"For task={task} there are no implemented Losses.")

    try:
        return loss_pool[loss_name](**loss_kwargs)
    except KeyError:
        raise ValueError(
            f'Given loss {loss_name} is not supported.'
            f'Please select from: {" | ".join(loss_pool.keys())}.'
        )
