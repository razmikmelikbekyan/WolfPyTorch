"""Package contains PIXEL LEVEL epochs for various task types."""
from .base import SingleBranchTrainEpoch, SingleBranchValidationEpoch
from .mixin import SimpleSingleChannelEpochMixin, PLMultiClassificationEpochMixin
from ...enums import TaskTypes


class PLRegressionTrainEpoch(SimpleSingleChannelEpochMixin, SingleBranchTrainEpoch):
    """Pixel level Regression Train Epoch"""


class PLRegressionValidationEpoch(SimpleSingleChannelEpochMixin, SingleBranchValidationEpoch):
    """Pixel levelRegression Validation Epoch"""


class PLBinaryClassificationTrainEpoch(SimpleSingleChannelEpochMixin, SingleBranchTrainEpoch):
    """Binary Classification Train Epoch"""


class PLBinaryClassificationValidationEpoch(SimpleSingleChannelEpochMixin, SingleBranchValidationEpoch):
    """Binary Classification Validation Epoch"""


class PLMultiClassificationTrainEpoch(PLMultiClassificationEpochMixin, SingleBranchTrainEpoch):
    """Multi Classification Train Epoch"""


class PLMultiClassificationValidationEpoch(PLMultiClassificationEpochMixin, SingleBranchValidationEpoch):
    """Multi Classification Validation Epoch"""


PIXEL_LEVEL_EPOCHS = {
    TaskTypes.PL_BINARY_CLASSIFICATION: (PLBinaryClassificationTrainEpoch, PLBinaryClassificationValidationEpoch),
    TaskTypes.PL_MULTI_CLASSIFICATION: (PLMultiClassificationTrainEpoch, PLMultiClassificationValidationEpoch),
    TaskTypes.PL_REGRESSION: (PLRegressionTrainEpoch, PLRegressionValidationEpoch)
}
