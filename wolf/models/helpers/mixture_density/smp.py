"""This package contains helper staff for modifying models from segmentation_models_pytorch to act like MDN models."""
from typing import Tuple, Dict

import torch
import torch.nn as nn
from segmentation_models_pytorch.base.model import SegmentationModel

from .mdn_heads import PLMDNHeadForSMP, MDNBottleneckHeadForSMP
from .base import split_mdn_output, PLMDNWrapper, BottleneckMDNWrapper


class PLMDNWrapperForSMP(PLMDNWrapper):
    """Mixture density network wrapper for the models from:
     https://github.com/qubvel/segmentation_models.pytorch

     It replaces the segmentation head with the MDNHeadForSMP.
     """

    def __init__(self, mixture_components: int = 1):
        super().__init__(mixture_components=mixture_components)
        self._mixture_components = mixture_components

        self.encoder = None
        self.decoder = None
        self.mdn_head = None

    def wrap_model(self, smp_model: SegmentationModel):
        """Wraps the model by replacing its segmentation head with MDNHead."""
        self.encoder = smp_model.encoder
        self.decoder = smp_model.decoder

        conv_layer, upsample_layer, _ = list(smp_model.segmentation_head)
        in_channels = conv_layer.in_channels
        kernel_size, _ = conv_layer.kernel_size
        upsampling = 1 if isinstance(upsample_layer, nn.Identity) else upsample_layer.scale_factor
        self.mdn_head = PLMDNHeadForSMP(in_channels, self.mixture_components, kernel_size, upsampling)
        return self

    def forward(self, x: torch.Tensor):
        return self.mdn_head(self.decoder(*self.encoder(x)))

    def split_mdn_output(self, mdn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits the MDN output into pi, mean and sigma."""
        assert len(mdn_output.shape) == 4, "MDN Output must have shape: B x C x H x W"
        return split_mdn_output(self.mixture_components, mdn_output)


class BottleneckMDNWrapperForSMP(BottleneckMDNWrapper):
    """Mixture density network wrapper for the models from:
     https://github.com/qubvel/segmentation_models.pytorch

     MDNBottleneckHeadForSMP raises from the bottleneck. Basically it becomes 2 path network.
     """

    def __init__(self, mixture_components: int = 1):
        super().__init__(mixture_components=mixture_components)
        self._mixture_components = mixture_components

        self.encoder = None
        self.decoder = None
        self.segmentation_head = None

        self.mdn_head = None

    def wrap_model(self, smp_model: SegmentationModel):
        """Wraps the model by adding MDN branch from the bottleneck."""
        self.encoder = smp_model.encoder
        self.decoder = smp_model.decoder
        self.segmentation_head = smp_model.segmentation_head

        in_channels = self.encoder.out_channels[-1]
        self.mdn_head = MDNBottleneckHeadForSMP(in_channels, self.mixture_components)
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns the outputs from 2 branches: segmentation and mdn from bottleneck."""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        segmentation_output = self.segmentation_head(decoder_output)
        mdn_output = self.mdn_head(features[-1])
        return {'pl_regression': segmentation_output, "tl_mdn": mdn_output}

    def split_mdn_output(self, mdn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits the MDN output into pi, mean and sigma."""
        assert mdn_output.ndim == 3, "MDN Output must hae shape: B x N-Components * 3 x 1"
        return split_mdn_output(self.mixture_components, mdn_output)
