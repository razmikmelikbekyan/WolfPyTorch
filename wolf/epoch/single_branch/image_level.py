"""Package contains IMAGE LEVEL epochs for various task types."""
from .base import SingleBranchTrainEpoch, SingleBranchValidationEpoch
from .mixin import SimpleSingleChannelEpochMixin

from ...enums import TaskTypes


class ILRegressionTrainEpoch(SimpleSingleChannelEpochMixin, SingleBranchTrainEpoch):
    """Regression Train Epoch"""


class ILRegressionValidationEpoch(SimpleSingleChannelEpochMixin, SingleBranchValidationEpoch):
    """Regression Validation Epoch"""


class ILBinaryClassificationTrainEpoch(SimpleSingleChannelEpochMixin, SingleBranchTrainEpoch):
    """Binary Classification Train Epoch"""


class ILBinaryClassificationValidationEpoch(SimpleSingleChannelEpochMixin, SingleBranchValidationEpoch):
    """Binary Classification Validation Epoch"""


class ILMultiClassificationTrainEpoch(SimpleSingleChannelEpochMixin, SingleBranchTrainEpoch):
    """Multi Classification Train Epoch"""


class ILMultiClassificationValidationEpoch(SimpleSingleChannelEpochMixin, SingleBranchValidationEpoch):
    """Multi Classification Validation Epoch"""


IMAGE_LEVEL_EPOCHS = {
    TaskTypes.IL_BINARY_CLASSIFICATION: (ILBinaryClassificationTrainEpoch, ILBinaryClassificationValidationEpoch),
    TaskTypes.IL_MULTI_CLASSIFICATION: (ILMultiClassificationTrainEpoch, ILMultiClassificationValidationEpoch),
    TaskTypes.IL_REGRESSION: (ILRegressionTrainEpoch, ILRegressionValidationEpoch)
}
