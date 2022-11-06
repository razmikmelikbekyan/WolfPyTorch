from typing import Dict, Any

import torch.nn as nn

from .helpers.weights.device import model_to_device
from .helpers.weights.loading import load_model_from_path
from .multi_branch import get_multi_branch_model
from .single_branch import get_single_branch_model
from ..enums import HighLevelArchitectureTypes, TaskTypes


def get_model(high_level_architecture: HighLevelArchitectureTypes,
              task: TaskTypes,
              model_name: str,
              **model_kwargs: Any) -> nn.Module:
    """Returns the loss class based on its name."""
    model_path = model_kwargs.pop("path", None)

    if high_level_architecture == HighLevelArchitectureTypes.SINGLE_BRANCH:
        model_obj = get_single_branch_model(task, model_name, **model_kwargs)
    elif high_level_architecture == HighLevelArchitectureTypes.MULTI_BRANCH:
        model_obj = get_multi_branch_model(task, model_name, **model_kwargs)
    else:
        raise NotImplementedError(f"For high level architecture={high_level_architecture} "
                                  f"there are on implemented models. ")

    model_obj = load_model_from_path(model_obj, model_path)
    return model_obj


def initialize_model(model_config: Dict, device: str = "cuda", multi_gpu: bool = True) -> nn.Module:
    """Initializes model from given path."""
    model = get_model(
        high_level_architecture=model_config['high_level_architecture'],
        task=model_config['task'],
        model_name=model_config['name'],
        **model_config.get('kwargs', {})
    )

    model = model_to_device(model, device=device, multi_gpu=multi_gpu)
    return model.eval()
