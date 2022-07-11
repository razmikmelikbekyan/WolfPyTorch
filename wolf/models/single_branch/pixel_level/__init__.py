"""This package SINGLE BRANCH architectures for PIXEL LEVEL tasks"""

from .fast_fcn import FastFCNVGG
from .fast_scnn import FastSCNN
from .fcn import FCNResnet
from .gated_scnn import GSCNN
from .smp import PixelLevelSMPModel, PixelLevelMDNSMPModel
from .transunet import TransUNet
from .unet import UNetVGG16, UNet
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
    'Unet': {
        'model_klass': UNet,
        'supported_tasks': (
            TaskTypes.PL_REGRESSION,
            TaskTypes.PL_BINARY_CLASSIFICATION,
            TaskTypes.PL_MULTI_CLASSIFICATION,
            TaskTypes.PL_QUANTILE_REGRESSION
        )

    },

    'UNetVGG16': {
        'model_klass': UNetVGG16,
        'supported_tasks': (
            TaskTypes.PL_REGRESSION,
            TaskTypes.PL_BINARY_CLASSIFICATION,
            TaskTypes.PL_MULTI_CLASSIFICATION,
            TaskTypes.PL_QUANTILE_REGRESSION
        )

    },

    "TransUnet": {
        'model_klass': TransUNet,
        'supported_tasks': (
            TaskTypes.PL_REGRESSION,
            TaskTypes.PL_BINARY_CLASSIFICATION,
            TaskTypes.PL_MULTI_CLASSIFICATION,
            TaskTypes.PL_QUANTILE_REGRESSION
        )

    },

    'FCNResnet': {
        'model_klass': FCNResnet,
        'supported_tasks': (
            TaskTypes.PL_REGRESSION,
            TaskTypes.PL_BINARY_CLASSIFICATION,
            TaskTypes.PL_MULTI_CLASSIFICATION,
            TaskTypes.PL_QUANTILE_REGRESSION
        )

    },

    "FastFCNVGG": {
        'model_klass': FastFCNVGG,
        'supported_tasks': (
            TaskTypes.PL_REGRESSION,
            TaskTypes.PL_BINARY_CLASSIFICATION,
            TaskTypes.PL_MULTI_CLASSIFICATION,
            TaskTypes.PL_QUANTILE_REGRESSION
        )

    },

    "FastSCNN": {
        'model_klass': FastSCNN,
        'supported_tasks': (
            TaskTypes.PL_REGRESSION,
            TaskTypes.PL_BINARY_CLASSIFICATION,
            TaskTypes.PL_MULTI_CLASSIFICATION,
            TaskTypes.PL_QUANTILE_REGRESSION
        )

    },

    "GSCNN": {
        'model_klass': GSCNN,
        'supported_tasks': (
            TaskTypes.PL_REGRESSION,
            TaskTypes.PL_BINARY_CLASSIFICATION,
            TaskTypes.PL_MULTI_CLASSIFICATION,
            TaskTypes.PL_QUANTILE_REGRESSION
        )

    },

}
