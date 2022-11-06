"""This package SINGLE BRANCH architectures for TILE LEVEL tasks"""
from .base import ImageLevelModel
from .mdn import ImageLevelMDNModel

from ....enums import TaskTypes

TILE_LEVEL_MODELS = {
    "ImageLevelModel": {
        'model_klass': ImageLevelModel,
        'supported_tasks': (
            TaskTypes.IL_REGRESSION,
            TaskTypes.IL_BINARY_CLASSIFICATION,
            TaskTypes.IL_MULTI_CLASSIFICATION,
            TaskTypes.IL_QUANTILE_REGRESSION
        )
    },
    "ImageLevelMDNModel": {
        "model_klass": ImageLevelMDNModel,
        "supported_tasks": (
            TaskTypes.IL_MDN_REGRESSION,
        )
    }
}
