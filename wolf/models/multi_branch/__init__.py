"""Package contains model that have MULTI BRANCH high level architecture."""
from typing import Any

import torch.nn as nn

from .smp import MultiBranchPLSimpleTLBottleneckMDNSMPModel
from ...constants import TaskTypes

MULTI_BRANCH_MODELS = {
    "MultiBranchPLSimpleTLBottleneckMDNSMPModel": {
        "model_klass": MultiBranchPLSimpleTLBottleneckMDNSMPModel,
        "supported_tasks": (TaskTypes.MULTI_TASK,)
    },
}


def get_multi_branch_model(task: TaskTypes, model_name: str, **model_kwargs: Any) -> nn.Module:
    """Returns a model instance."""

    try:
        model_klass = MULTI_BRANCH_MODELS[model_name]['model_klass']
    except KeyError:
        raise ValueError(
            f'Given model={model_name} is not supported.'
            f'Please select from: {" | ".join(MULTI_BRANCH_MODELS.keys())}.'
        )

    if task not in MULTI_BRANCH_MODELS[model_name]['supported_tasks']:
        raise ValueError(f"Given model={model_name} is not supporting given task={task}")

    model_obj = model_klass(**model_kwargs)
    return model_obj
