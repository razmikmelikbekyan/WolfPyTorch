from typing import Tuple, Any

import torch
from torch import nn as nn
from torch.nn import functional as F

from .base import ImageLevelModel
from ...helpers.mixture_density.commons import join_mdn_output, split_mdn_output


class TLMDNHead(nn.Module):
    """Special Head that takes model features as an input."""

    def __init__(self, in_features: int, mixture_components: int):
        super().__init__()

        self.fcn_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 512)
        )

        self.pi = nn.Linear(512, mixture_components)
        self.mean = nn.Linear(512, mixture_components)
        self.sigma = nn.Linear(512, mixture_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the input through MDN layers -> pi - mean - sigma."""
        x = self.fcn_layers(x)
        x = join_mdn_output(F.softmax(self.pi(x), dim=1), self.mean(x), torch.exp(self.sigma(x)))
        return x


class ImageLevelMDNModel(ImageLevelModel):
    """Simple tile level MDN model.

    It supports the following task types:
        - image level MDN regression
    """

    def __init__(self, *args, mixture_components: int, **kwargs: Any):
        kwargs['out_channels'] = 0
        super().__init__(*args, **kwargs)
        self.mixture_components = mixture_components
        self.mdn_head = TLMDNHead(self.n_features, mixture_components)

    def forward(self, x):
        return self.mdn_head(self.timm_model.forward(x))

    def split_mdn_output(self, mdn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits the MDN output into pi, mean and sigma."""
        assert mdn_output.ndim == 2, f"MDN Output must hae shape: B x N-Components * 3, given={mdn_output.shape}"
        return split_mdn_output(self.mixture_components, mdn_output)
