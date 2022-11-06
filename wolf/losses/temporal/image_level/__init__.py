"""Package contains temporal IMAGE LEVEL losses for various task types."""
from .classification import *

from ....enums import TaskTypes

TEMPORAL_IMAGE_LEVEL_LOSSES = {
    TaskTypes.IL_MULTI_CLASSIFICATION: {
        'TemporalCrossEntropyLoss': TemporalCrossEntropyLoss,
        'TemporalMultiClassFocalLoss': TemporalMultiClassFocalLoss,
    },
}
