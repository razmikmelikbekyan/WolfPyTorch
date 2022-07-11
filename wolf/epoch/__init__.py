from typing import Tuple, Type

from .multi_branch import MultiBranchEpochMixin, MultiBranchTrainEpoch, MultiBranchValidationEpoch
from .single_branch import SingleBranchEpochMixin, SingleBranchTrainEpoch, SingleBranchValidationEpoch
from .single_branch import get_single_branch_epoch
from .temporal import TemporalEpochMixin
from ..constants import TaskTypes, HighLevelArchitectureTypes


def get_epochs(high_level_architecture: HighLevelArchitectureTypes, task: TaskTypes,
               ) -> Tuple[Type[SingleBranchTrainEpoch], Type[SingleBranchValidationEpoch]]:
    """Returns the Epoch classes based on given task."""
    if high_level_architecture == HighLevelArchitectureTypes.SINGLE_BRANCH:
        epochs = get_single_branch_epoch(task)
    else:
        raise NotImplementedError(f"For high level architecture={high_level_architecture} "
                                  f"there are on implemented epochs. ")

    return epochs
