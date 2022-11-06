"""Package contains IMAGE LEVEL losses for various task types."""
from .classification import *
from .regression import *

from ....enums import TaskTypes

IMAGE_LEVEL_LOSSES = {
    TaskTypes.IL_BINARY_CLASSIFICATION: {
        'BCELoss': BCELoss,
        'BCEWithLogitsLoss': BCEWithLogitsLoss,
        'BinaryFocalLoss': BinaryFocalLoss
    },
    TaskTypes.IL_MULTI_CLASSIFICATION: {
        'CrossEntropyLoss': CrossEntropyLoss,
        'NLLLoss': NLLLoss,
        'LabelSmoothingLoss': LabelSmoothingLoss,
        'MultiClassFocalLoss': MultiClassFocalLoss,
    },
    TaskTypes.IL_REGRESSION: {
        'L1Loss': L1Loss,
        'SmoothL1Loss': SmoothL1Loss,
        'L2Loss': L2Loss,
        'HuberLoss': HuberLoss,
    },
    TaskTypes.IL_QUANTILE_REGRESSION: {
        'PinBallLoss': PinBallLoss
    },
    TaskTypes.IL_MDN_REGRESSION: {
        'MDNLoss': MDNLoss
    }
}
