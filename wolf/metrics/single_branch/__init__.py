from typing import Any

from .pixel_level import PIXEL_LEVEL_EVALUATORS
from .image_level import IMAGE_LEVEL_EVALUATORS
from ..base import BaseEvaluator
from ...enums import TaskTypes


def get_single_branch_evaluator(task: TaskTypes, **evaluator_kwargs: Any) -> BaseEvaluator:
    """Returns the evaluator class based on its task."""

    if task in TaskTypes.get_image_level_tasks():
        evaluator_klass = IMAGE_LEVEL_EVALUATORS[task]
    elif task in TaskTypes.get_pixel_level_tasks():
        evaluator_klass = PIXEL_LEVEL_EVALUATORS[task]
    else:
        raise NotImplementedError(f"For task={task} there is no implemented Evaluator.")

    return evaluator_klass(**evaluator_kwargs)
