from typing import Dict

import torch

from .base import Loss
from .pixel_level import PIXEL_LEVEL_LOSSES
from .image_level import IMAGE_LEVEL_LOSSES
from ...enums import TaskTypes


def get_single_branch_loss(task: TaskTypes, loss_name: str, **loss_kwargs: Dict) -> Loss:
    """Returns the loss class based on its name."""
    loss_kwargs = {
        k: (torch.tensor(v, device='cpu') if isinstance(v, list) else v)
        for k, v in loss_kwargs.items()
    }

    if task in TaskTypes.get_image_level_tasks():
        loss_pool = IMAGE_LEVEL_LOSSES[task]
    elif task in TaskTypes.get_pixel_level_tasks():
        loss_pool = PIXEL_LEVEL_LOSSES[task]
    else:
        raise NotImplementedError(f"For task={task} there are no implemented Losses.")

    try:
        return loss_pool[loss_name](**loss_kwargs)
    except KeyError:
        raise ValueError(
            f'Given loss {loss_name} is not supported for task={task}.'
            f'Please select from: {" | ".join(loss_pool.keys())}.'
        )
