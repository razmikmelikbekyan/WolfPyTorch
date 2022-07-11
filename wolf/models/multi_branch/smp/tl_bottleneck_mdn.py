from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...helpers.mixture_density.commons import join_mdn_output, split_mdn_output
from ...single_branch.pixel_level.smp import PixelLevelSMPModel


class TLMDNBottleneckHeadForSMP(nn.Module):
    """Special tile level head that arises from the SMP Model bottleneck."""
    _pooled_image_size = 4

    def __init__(self, in_channels: int, mixture_components: int):
        super().__init__()

        self.pooling_layer = nn.AdaptiveAvgPool2d(self._pooled_image_size)
        self.fcn_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),  # TODO: get from config
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
        x = join_mdn_output(F.softmax(self.pi(x), dim=1), self.mean(x), torch.exp(self.sigma(x)))
        return x


class MultiBranchPLSimpleTLBottleneckMDNSMPModel(PixelLevelSMPModel):
    """Multi branch SMP based model.

    The first branch outputs a Pixel Level result that may include on of the following tasks:
        - pixel level regression
        - pixel level binary classification
        - pixel level multi classification
        - pixel level quantile regression

    The second branch outputs a Tile Level result for the following tasks:
        - tile level MDN regression
    """
    PL_BRANCH_NAME = "_PL_SIMPLE_"
    TL_BRANCH_NAME = "_TL_MDN_"

    def __init__(self, *args, mixture_components: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixture_components = mixture_components
        self.mdn_head = self.get_mdn_head()

    def get_mdn_head(self):
        in_channels = self.smp_model.encoder.out_channels[-1]
        return TLMDNBottleneckHeadForSMP(in_channels, self.mixture_components)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns the outputs from 2 branches: simple pixel level and mdn from bottleneck."""
        features = self.smp_model.encoder(x)
        decoder_output = self.smp_model.decoder(*features)
        pl_output = self.smp_model.segmentation_head(decoder_output)
        mdn_output = self.mdn_head(features[-1])
        return {self.PL_BRANCH_NAME: pl_output, self.TL_BRANCH_NAME: mdn_output}

    def split_mdn_output(self, mdn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits the MDN output into pi, mean and sigma."""
        assert mdn_output.ndim == 2, f"MDN Output must hae shape: B x N-Components * 3, given={mdn_output.shape}"
        return split_mdn_output(self.mixture_components, mdn_output)
