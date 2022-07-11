import torch
from segmentation_models_pytorch.base import SegmentationHead
from torch import nn as nn
from torch.nn import functional as F


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
        return torch.cat([self.pi(x), self.mean(x), torch.exp(self.sigma(x))], dim=1)


class MDNBottleneckHeadForSMP(nn.Module):
    """Special Head that arises from the Model bottleneck."""
    _pooled_image_size = 4

    def __init__(self, in_channels: int, mixture_components: int):
        super().__init__()

        self.pooling_layer = nn.AdaptiveAvgPool2d(self._pooled_image_size)
        self.fcn_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear((in_channels * self._pooled_image_size * self._pooled_image_size), 2048),
            nn.Linear(2048, 512)
        )

        self.pi = nn.Linear(512, mixture_components)
        self.mean = nn.Linear(512, mixture_components)
        self.sigma = nn.Linear(512, mixture_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the input through MDN layers -> pi - mean - sigma."""
        x = self.pooling_layer(x)
        x = self.fcn_layers(x)
        x = torch.cat([F.softmax(self.pi(x), dim=1), self.mean(x), torch.exp(self.sigma(x))], dim=1)
        return x.unsqueeze(dim=-1)
