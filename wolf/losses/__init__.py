from typing import Any

from .multi_branch import get_multi_branch_loss, MultiBranchLoss, MULTI_BRANCH_LOSS_CONFIG_KEYS
from .single_branch import get_single_branch_loss, Loss
from .temporal import get_temporal_loss, TemporalLoss
from ..enums import TaskTypes, HighLevelArchitectureTypes


def get_loss(high_level_architecture: HighLevelArchitectureTypes,
             task: TaskTypes,
             loss_name: str,
             **loss_kwargs: Any) -> Loss:
    """Returns the loss class based on its name."""
    if high_level_architecture == HighLevelArchitectureTypes.SINGLE_BRANCH:
        loss_obj = get_single_branch_loss(task, loss_name, **loss_kwargs)
    elif high_level_architecture == HighLevelArchitectureTypes.MULTI_BRANCH:
        loss_obj = get_multi_branch_loss(task, loss_name, **loss_kwargs)
    elif high_level_architecture == HighLevelArchitectureTypes.TEMPORAL:
        loss_obj = get_temporal_loss(task, loss_name, **loss_kwargs)

    else:
        raise NotImplementedError(f"For high level architecture={high_level_architecture} "
                                  f"there are on implemented losses. ")

    return loss_obj
