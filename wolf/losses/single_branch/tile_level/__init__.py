"""Package contains TILE LEVEL losses for various task types."""
from .classification import *
from .regression import *

from ....constants import TaskTypes

TILE_LEVEL_LOSSES = {
    TaskTypes.TL_BINARY_CLASSIFICATION: {
        'BCELoss': BCELoss,
        'BCEWithLogitsLoss': BCEWithLogitsLoss,
        'BinaryFocalLoss': BinaryFocalLoss
    },
    TaskTypes.TL_MULTI_CLASSIFICATION: {
        'CrossEntropyLoss': CrossEntropyLoss,
        'NLLLoss': NLLLoss,
        'LabelSmoothingLoss': LabelSmoothingLoss,
        'MultiClassFocalLoss': MultiClassFocalLoss,
    },
    TaskTypes.TL_REGRESSION: {
        'L1Loss': L1Loss,
        'SmoothL1Loss': SmoothL1Loss,
        'L2Loss': L2Loss,
        'HuberLoss': HuberLoss,
    },
    TaskTypes.TL_QUANTILE_REGRESSION: {
        'PinBallLoss': PinBallLoss
    },
    TaskTypes.TL_MDN_REGRESSION: {
        'MDNLoss': MDNLoss
    }
}
