"""Package contains model that have SINGLE BRANCH high level architecture."""
from typing import Any

import torch.nn as nn

from .pixel_level import PIXEL_LEVEL_MODELS
from .tile_level import TILE_LEVEL_MODELS
from ...enums import TaskTypes


def get_single_branch_model(task: TaskTypes, model_name: str, **model_kwargs: Any) -> nn.Module:
    """Returns a model instance."""
    if task in TaskTypes.get_tile_level_tasks():
        models_pool = TILE_LEVEL_MODELS
    elif task in TaskTypes.get_pixel_level_tasks():
        models_pool = PIXEL_LEVEL_MODELS
    else:
        raise NotImplementedError(f"For task={task} there are no implemented models.")

    try:
        model_klass = models_pool[model_name]['model_klass']
    except KeyError:
        raise ValueError(
            f'Given model={model_name} is not supported.'
            f'Please select from: {" | ".join(models_pool.keys())}.'
        )

    if task not in models_pool[model_name]['supported_tasks']:
        raise ValueError(f"Given model={model_name} is not supporting given task={task}")

    model_obj = model_klass(**model_kwargs)
    return model_obj
