from typing import Any

from .base import BaseEvaluator
from .meter import AverageValueMeter
from .multi_branch import get_multi_branch_evaluator, MultiBranchEvaluator, MULTI_BRANCH_EVALUATOR_CONFIG_KEYS
from .single_branch import get_single_branch_evaluator, BaseEvaluator
from ..constants import TaskTypes, HighLevelArchitectureTypes


def get_evaluator(high_level_architecture: HighLevelArchitectureTypes,
                  task: TaskTypes,
                  **evaluator_kwargs: Any) -> BaseEvaluator:
    """Returns the evaluator object based on its task."""
    if high_level_architecture == HighLevelArchitectureTypes.SINGLE_BRANCH:
        evaluator_obj = get_single_branch_evaluator(task, **evaluator_kwargs)
    elif high_level_architecture == HighLevelArchitectureTypes.MULTI_BRANCH:
        evaluator_obj = get_multi_branch_evaluator(task, **evaluator_kwargs)
    else:
        raise NotImplementedError(f"For high level architecture={high_level_architecture} "
                                  f"there are no implemented evaluators. ")
    return evaluator_obj
