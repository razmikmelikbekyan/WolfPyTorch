"""Package contains TILE LEVEL losses for various task types."""
from .classification import *

from ....enums import TaskTypes

TILE_LEVEL_LOSSES = {
    TaskTypes.TL_MULTI_CLASSIFICATION: {
        'TemporalCrossEntropyLoss': TemporalCrossEntropyLoss,
        'TemporalMultiClassFocalLoss': TemporalMultiClassFocalLoss,
    },
}
