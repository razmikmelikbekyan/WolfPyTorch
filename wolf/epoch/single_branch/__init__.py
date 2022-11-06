"""Package contains epoch objects for high level architecture SINGLE BRANCH that are responsible for various tasks."""

from .base import SingleBranchEpochMixin, SingleBranchTrainEpoch, SingleBranchValidationEpoch
from .image_level import IMAGE_LEVEL_EPOCHS
from .pixel_level import PIXEL_LEVEL_EPOCHS
from ...enums import TaskTypes


def get_single_branch_epoch(task: TaskTypes) -> (SingleBranchTrainEpoch, SingleBranchValidationEpoch):
    """Returns a model instance."""
    if task in TaskTypes.get_image_level_tasks():
        epochs_pool = IMAGE_LEVEL_EPOCHS
    elif task in TaskTypes.get_pixel_level_tasks():
        epochs_pool = PIXEL_LEVEL_EPOCHS
    else:
        raise NotImplementedError(f"For task={task} there are no implemented models.")

    try:
        epochs = epochs_pool[task]
    except KeyError:
        raise ValueError(
            f'Given task={task} is not supported.'
            f'Please select from: {" | ".join(epochs_pool.keys())}.'
        )
    return epochs
