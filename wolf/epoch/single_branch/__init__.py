"""Package contains epoch objects for high level architecture SINGLE BRANCH that are responsible for various tasks."""

from .base import SingleBranchEpochMixin, SingleBranchTrainEpoch, SingleBranchValidationEpoch
from .pixel_level import PIXEL_LEVEL_EPOCHS
from .tile_level import TILE_LEVEL_EPOCHS
from ...constants import TaskTypes


def get_single_branch_epoch(task: TaskTypes) -> (SingleBranchTrainEpoch, SingleBranchValidationEpoch):
    """Returns a model instance."""
    if task in TaskTypes.get_tile_level_tasks():
        epochs_pool = TILE_LEVEL_EPOCHS
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
