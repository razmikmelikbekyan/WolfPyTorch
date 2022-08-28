from typing import Tuple

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead

from .base import PixelLevelSMPModel
from ...helpers.mixture_density.commons import join_mdn_output, split_mdn_output


class PLMDNHeadForSMP(nn.Module):
    """Special Head that replaces the Segmentation head."""

    def __init__(self, in_channels: int, mixture_components: int, kernel_size: int, upsampling: int):
        super().__init__()

        self.pi = SegmentationHead(in_channels, mixture_components,
                                   activation="softmax2d", kernel_size=kernel_size, upsampling=upsampling)
        self.mean = SegmentationHead(in_channels, mixture_components,
                                     activation=None, kernel_size=kernel_size, upsampling=upsampling)
        self.sigma = SegmentationHead(in_channels, mixture_components,
                                      activation=None, kernel_size=kernel_size, upsampling=upsampling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the input through MDN layers -> pi - mean - sigma."""
        return join_mdn_output(self.pi(x), self.mean(x), torch.exp(self.sigma(x)))


class PixelLevelMDNSMPModel(PixelLevelSMPModel):
    """Simple pixel level MDN model.

    It supports the following task types:
        - pixel level MDN regression
    """

    def __init__(self, *args, mixture_components: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixture_components = mixture_components
        self.mdn_head = self.get_mdn_head()

    def get_mdn_head(self):
        conv_layer, upsample_layer, _ = list(self.smp_model.segmentation_head)
        in_channels = conv_layer.in_channels
        kernel_size, _ = conv_layer.kernel_size
        upsampling = 1 if isinstance(upsample_layer, nn.Identity) else upsample_layer.scale_factor
        return PLMDNHeadForSMP(in_channels, self.mixture_components, kernel_size, upsampling)

    def forward(self, x):
        return self.mdn_head(self.smp_model.decoder(*self.smp_model.encoder(x)))

    def split_mdn_output(self, mdn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits the MDN output into pi, mean and sigma."""
        assert len(mdn_output.shape) == 4, f"MDN Output must have shape: B x C x H x W, given={mdn_output.shape}"
        return split_mdn_output(self.mixture_components, mdn_output)
