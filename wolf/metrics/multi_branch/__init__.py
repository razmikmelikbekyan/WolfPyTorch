from typing import List, Dict

from .base import MultiBranchEvaluator
from ..single_branch import get_single_branch_evaluator
from ...enums import TaskTypes

MULTI_BRANCH_EVALUATOR_CONFIG_KEYS = ('task', 'branch', 'evaluator_kwargs')


def get_multi_branch_evaluator(task: TaskTypes,
                               multibranch_evaluator_config: List[Dict],
                               device: str) -> MultiBranchEvaluator:
    """Returns the multibranch evaluator."""
    assert task == TaskTypes.MULTI_TASK, task

    message = f"All items of config must be dict with the following keys: {MULTI_BRANCH_EVALUATOR_CONFIG_KEYS}"
    assert all(set(x.keys()) == set(MULTI_BRANCH_EVALUATOR_CONFIG_KEYS) for x in multibranch_evaluator_config), message

    branches_evaluators = {
        item['branch']: get_single_branch_evaluator(item['task'], **item['evaluator_kwargs'], device=device)
        for item in multibranch_evaluator_config
    }
    return MultiBranchEvaluator(branches_evaluators)
