"""Package contains PIXEL LEVEL losses for various task types."""
from .classification import *
from .regression import *

from ....constants import TaskTypes

PIXEL_LEVEL_LOSSES = {
    TaskTypes.PL_BINARY_CLASSIFICATION: {
        'BCELoss': BCELoss,
        'BCEWithLogitsLoss': BCEWithLogitsLoss,
        'BinaryFocalLoss': BinaryFocalLoss,
        'BinaryJaccardLoss': BinaryJaccardLoss,
        'BinaryDiceLoss': BinaryDiceLoss,
        'BinaryTverskyLoss': BinaryTverskyLoss,
        'BCEDiceLoss': BCEDiceLoss,
        'BCEDiceLogitsLoss': BCEDiceLogitsLoss,
    },
    TaskTypes.PL_MULTI_CLASSIFICATION: {
        'CrossEntropyLoss': CrossEntropyLoss,
        'MultiClassJaccardLoss': MultiClassJaccardLoss,
        'MultiClassDiceLoss': MultiClassDiceLoss,
        'MultiClassFocalLoss': MultiClassFocalLoss,
        'MultiClassTverskyLoss': MultiClassTverskyLoss,
    },
    TaskTypes.PL_REGRESSION: {
        'L1Loss': L1Loss,
        'SmoothL1Loss': SmoothL1Loss,
        'L2Loss': L2Loss
    },
    TaskTypes.PL_QUANTILE_REGRESSION: {
        'PinBallLoss': PinBallLoss,
        'L2PinballLoss': L2PinballLoss
    },
    TaskTypes.PL_MDN_REGRESSION: {
        'MDNLoss': MDNLoss
    }
}
