from typing import Any

from .base import BaseEvaluator
from .pixel_level import PIXEL_LEVEL_EVALUATORS
from .tile_level import TILE_LEVEL_EVALUATORS
from ...constants import TaskTypes


def get_single_branch_evaluator(task: TaskTypes, **evaluator_kwargs: Any) -> BaseEvaluator:
    """Returns the evaluator class based on its task."""

    if task in TaskTypes.get_tile_level_tasks():
        evaluator_klass = TILE_LEVEL_EVALUATORS[task]
    elif task in TaskTypes.get_pixel_level_tasks():
        evaluator_klass = PIXEL_LEVEL_EVALUATORS[task]
    else:
        raise NotImplementedError(f"For task={task} there is no implemented Evaluator.")

    return evaluator_klass(**evaluator_kwargs)
