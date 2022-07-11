from typing import List, Dict

from .base import MultiBranchLoss
from ..single_branch import get_single_branch_loss
from ...constants import TaskTypes

MULTI_BRANCH_LOSS_CONFIG_KEYS = ('task', 'branch', 'branch_weight', 'loss_name', 'loss_kwargs')


def get_multi_branch_loss(task: TaskTypes, loss_name: str, multibranch_loss_config: List[Dict]) -> MultiBranchLoss:
    """Returns the multibranch loss."""
    assert loss_name == 'MultiBranchLoss', loss_name
    assert task == TaskTypes.MULTI_TASK, task

    message = f"All items of config must be dict with the following keys: {MULTI_BRANCH_LOSS_CONFIG_KEYS}"
    assert all(set(x.keys()) == set(MULTI_BRANCH_LOSS_CONFIG_KEYS) for x in multibranch_loss_config), message

    branches_losses = {
        item['branch']: get_single_branch_loss(item['task'], item['loss_name'], **item['loss_kwargs'])
        for item in multibranch_loss_config
    }
    branches_weights = {
        item['branch']: item['branch_weight'] for item in multibranch_loss_config
    }
    return MultiBranchLoss(branches_losses, branches_weights)
