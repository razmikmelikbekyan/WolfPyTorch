"""This package SINGLE BRANCH architectures for TILE LEVEL tasks"""
from .base import TileLevelModel
from .mdn import TileLevelMDNModel

from ....enums import TaskTypes

TILE_LEVEL_MODELS = {
    "TileLevelModel": {
        'model_klass': TileLevelModel,
        'supported_tasks': (
            TaskTypes.TL_REGRESSION,
            TaskTypes.TL_BINARY_CLASSIFICATION,
            TaskTypes.TL_MULTI_CLASSIFICATION,
            TaskTypes.TL_QUANTILE_REGRESSION
        )
    },
    "TileLevelMDNModel": {
        "model_klass": TileLevelMDNModel,
        "supported_tasks": (
            TaskTypes.TL_MDN_REGRESSION,
        )
    }
}
