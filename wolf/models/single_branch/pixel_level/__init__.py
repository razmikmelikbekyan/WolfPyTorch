"""This package SINGLE BRANCH architectures for PIXEL LEVEL tasks"""

from .smp import PixelLevelSMPModel
from .mdn import PixelLevelMDNSMPModel
from ....constants import TaskTypes

PIXEL_LEVEL_MODELS = {
    "PixelLevelSMPModel": {
        'model_klass': PixelLevelSMPModel,
        'supported_tasks': (
            TaskTypes.PL_REGRESSION,
            TaskTypes.PL_BINARY_CLASSIFICATION,
            TaskTypes.PL_MULTI_CLASSIFICATION,
            TaskTypes.PL_QUANTILE_REGRESSION
        )

    },
    "PixelLevelMDNSMPModel": {
        'model_klass': PixelLevelMDNSMPModel,
        'supported_tasks': (
            TaskTypes.PL_MDN_REGRESSION,
        )

    },
}
