"""Package contains TILE LEVEL epochs for various task types."""
from .base import SingleBranchTrainEpoch, SingleBranchValidationEpoch
from .mixin import SimpleSingleChannelEpochMixin

from ...enums import TaskTypes


class TLRegressionTrainEpoch(SimpleSingleChannelEpochMixin, SingleBranchTrainEpoch):
    """Regression Train Epoch"""


class TLRegressionValidationEpoch(SimpleSingleChannelEpochMixin, SingleBranchValidationEpoch):
    """Regression Validation Epoch"""


class TLBinaryClassificationTrainEpoch(SimpleSingleChannelEpochMixin, SingleBranchTrainEpoch):
    """Binary Classification Train Epoch"""


class TLBinaryClassificationValidationEpoch(SimpleSingleChannelEpochMixin, SingleBranchValidationEpoch):
    """Binary Classification Validation Epoch"""


class TLMultiClassificationTrainEpoch(SimpleSingleChannelEpochMixin, SingleBranchTrainEpoch):
    """Multi Classification Train Epoch"""


class TLMultiClassificationValidationEpoch(SimpleSingleChannelEpochMixin, SingleBranchValidationEpoch):
    """Multi Classification Validation Epoch"""


TILE_LEVEL_EPOCHS = {
    TaskTypes.TL_BINARY_CLASSIFICATION: (TLBinaryClassificationTrainEpoch, TLBinaryClassificationValidationEpoch),
    TaskTypes.TL_MULTI_CLASSIFICATION: (TLMultiClassificationTrainEpoch, TLMultiClassificationValidationEpoch),
    TaskTypes.TL_REGRESSION: (TLRegressionTrainEpoch, TLRegressionValidationEpoch)
}
