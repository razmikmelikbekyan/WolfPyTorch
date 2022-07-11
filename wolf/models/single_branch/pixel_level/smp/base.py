from typing import Optional, Any

import torch
import torch.nn as nn
from segmentation_models_pytorch import Unet, Linknet, UnetPlusPlus, MAnet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus

SMP_MODELS = {
    'Unet': Unet,
    'UnetPlusPlus': UnetPlusPlus,
    'MAnet': MAnet,
    'FPN': FPN,
    'PAN': PAN,
    'Linknet': Linknet,
    'PSPNet': PSPNet,
    'DeepLabV3': DeepLabV3,
    'DeepLabV3Plus': DeepLabV3Plus
}


class PixelLevelSMPModel(nn.Module):
    """Special wrapper class for all models from "segmentation_models_pytorch" package.

    It supports the following task types:
        - pixel level regression
        - pixel level binary classification
        - pixel level multi classification
        - pixel level quantile regression
    """

    def __init__(self, architecture: str, **architecture_kwargs: Optional[Any]):
        """

        Args:
            architecture: the architecture name, like UNet, FPN and so on
            **architecture_kwargs: the architecture arguments
        """
        super().__init__()
        try:
            model_klass = SMP_MODELS[architecture]
        except KeyError:
            raise ValueError(
                f'Given model {architecture} is not supported by "segmentation_models_pytorch" package.'
                f'Please select from: {" | ".join(SMP_MODELS.keys())}.'
            )
        self.smp_model = model_klass(**architecture_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.smp_model.forward(x)
