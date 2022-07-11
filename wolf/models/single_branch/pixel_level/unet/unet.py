import math
from typing import Tuple

import torch
from torch import nn

from ....helpers.layers.cnn import ConvBNActivation, CenterDilation


class UNetConvBlock(nn.Module):
    """
    A helper Module that performs 2 convolutions.
    A ReLU activation follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding: bool,
                 batch_norm: bool,
                 pooling: bool,
                 dropout_rate: float):
        super(UNetConvBlock, self).__init__()

        params = {
            'kernel_size': 3,
            'padding': (3 // 2) if padding else 0,
            'stride': 1,
            'dilation': 1,

        }

        self.block = nn.Sequential(
            ConvBNActivation(in_channels, out_channels, batch_norm=batch_norm, dropout_rate=dropout_rate, **params),
            ConvBNActivation(out_channels, out_channels, batch_norm=batch_norm, dropout_rate=dropout_rate, **params),
        )

        if pooling:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pooling = None

    def forward(self, x: torch.Tensor):
        x = self.block(x)

        before_pooling = x
        if self.pooling is not None:
            x = self.pooling(x)

        return x, before_pooling


class UNetUpBlock(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 upsampling_type: str,
                 upsampling_mode: str,
                 padding: bool,
                 batch_norm: bool,
                 dropout_rate: float):
        super(UNetUpBlock, self).__init__()
        if upsampling_type == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode=upsampling_mode, scale_factor=2, align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        final_in_channels = skip_channels + out_channels
        self.conv_block = UNetConvBlock(final_in_channels, out_channels, padding, batch_norm, False, dropout_rate)

    @staticmethod
    def center_crop(x: torch.Tensor, target_size: Tuple[int, int]):
        target_h, target_w = target_size
        _, _, h, w = x.size()
        diff_h = (h - target_h) // 2
        diff_w = (w - target_w) // 2
        return x[:, :, diff_h: (diff_h + target_h), diff_w: (diff_w + target_w)]

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor):
        x = self.up(x)
        x = torch.cat((x, self.center_crop(skip_connection, x.shape[2:])), dim=1)
        x, _ = self.conv_block(x)
        return x


class UNet(nn.Module):
    """
    Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015) https://arxiv.org/abs/1505.04597
    Using the default arguments will yield the exact version used in the original paper.
    """

    UPSAMPLING_TYPES = {
        'upconv': nn.ConvTranspose2d,
        'upsample': nn.Upsample
    }

    UPSAMPLING_MODES = (
        'nearest',
        'linear',
        'bilinear',
        'bicubic',
    )

    def __init__(
            self,
            in_channels: int = 3,
            classes: int = 1,
            depth: int = 5,
            init_features: int = 64,
            padding: bool = False,
            batch_norm: bool = False,
            upsampling_type: str = 'upconv',
            upsampleing_mode: str = 'bilinear',
            dropout_rate: float = 0,
            dilation_depth: int = 0,
            dilation_type: str = 'cascade',
    ):
        """
        Args:
            in_channels (int): number of input channels
            classes (int): number of output channels or classes
            depth (int): depth of the network
            init_features (int): number of filters in the first layer
            padding (bool): if True, apply padding such that the input shape is the same as the output,
                            this may introduce artifacts
            batch_norm (bool): use BatchNorm after layers with an activation function,
            upsampling_type (str): one of 'upconv' or 'upsample'.
                            'upconv' will use transposed convolutions for learned upsampling.
                            'upsample' will use bilinear upsampling.
            upsampling_type (str): one of 'nearest', 'linear', 'bilinear', 'bicubic', defines the algorithm to use
                                   for UpSample layers
            dropout_rate: (float): if >0 will apply some dropout in encoder part of the network
            dilation_depth: if > 0, will apply dilated convolutions in bottleneck of network
            dilation_type: one of 'cascade' or 'parallel', defines the dilation layer type
        """
        if upsampling_type not in self.UPSAMPLING_TYPES:
            raise ValueError(
                f'upsampling_type must be one of the followings {list(self.UPSAMPLING_TYPES.keys())}, '
                f'got: "{upsampling_type}" .'
            )

        if upsampleing_mode not in self.UPSAMPLING_MODES:
            raise ValueError(
                f'upsampleing_mode must be one of the followings {self.UPSAMPLING_MODES}, '
                f'got: "{upsampleing_mode}" .'
            )

        super(UNet, self).__init__()

        self.padding = padding
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.dilation_type = dilation_type

        self.encoders = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                current_in, current_out = in_channels, init_features * (2 ** i)
            else:
                current_in, current_out = init_features * (2 ** (i - 1)), init_features * (2 ** i)

            pooling = i < depth - 1
            self.encoders.append(UNetConvBlock(current_in, current_out, padding, batch_norm, pooling, dropout_rate))

        if dilation_depth:
            current_channels = init_features * (2 ** (depth - 1))
            self.center_dilation = CenterDilation(current_channels, dilation_depth, batch_norm, dilation_type)
        else:
            self.center_dilation = None

        self.decoders = nn.ModuleList()
        for i in reversed(range(1, depth)):
            current_in, current_out = init_features * (2 ** i), init_features * (2 ** (i - 1))
            self.decoders.append(
                UNetUpBlock(
                    current_in, current_out, current_out,
                    upsampling_type, upsampleing_mode, padding, batch_norm, dropout_rate=0,
                )
            )

        self.classifier = nn.Sequential(
            nn.Conv2d(init_features, classes, kernel_size=1, bias=False),
        )

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                # nn.init.normal_(m.weight.data, 1.0, 0.02)
                # nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        skip_blocks = []
        for i, encoder in enumerate(self.encoders):
            x, before_pool = encoder(x)
            if i != self.depth - 1:
                skip_blocks.append(before_pool)  # all, except the last one

        if self.center_dilation is not None:
            x = self.center_dilation(x)

        for decoder, skip_block in zip(self.decoders, skip_blocks[::-1]):
            x = decoder(x, skip_block)

        return self.classifier(x)
