import torch
import torch.nn as nn
import torchvision

from ....helpers.layers.cnn import CenterDilation
from ....helpers.weights.adjust_first_conv import patch_first_conv


class UNetVGG16DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            middle_channels: int,
            out_channels: int,
            upsampling_type: str,
    ):
        super().__init__()
        self.in_channels = in_channels

        if upsampling_type == 'upconv':
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(mode="upsampling_mode", scale_factor=2, align_corners=True),
                nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetVGG16(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            classes: int = 1,
            init_features: int = 32,
            pretrained: bool = False,
            upsampling_type: str = 'upconv',
            dilation_depth: int = 0,
            dilation_type: str = 'cascade'

    ):
        """
        Args:
            in_channels (int): number of input channels
            classes (int): number of output channels or classes
            init_features (int): number of filters in the first layer
            pretrained (bool): False - no pre-trained network used, True - encoder pre-trained with VGG16
            upsampling_type (str): one of 'upconv' or 'upsample'.
                            'upconv' will use transposed convolutions for learned upsampling.
                            'upsample' will use bilinear upsampling.
            dilation_depth: if > 0, will apply dilated convolutions in bottleneck of network
            dilation_type: one of 'cascade' or 'parallel', defines the dilation layer type
        """

        super().__init__()
        self.dilation_type = dilation_type

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features
        patch_first_conv(self.encoder, in_channels)
        for layer in self.encoder.children():
            if isinstance(layer, nn.Conv2d):
                layer.bias = None

        self.conv1 = nn.Sequential(
            self.encoder[0],
            self.relu,
            self.encoder[2],
            self.relu
        )

        self.conv2 = nn.Sequential(
            self.encoder[5],
            self.relu,
            self.encoder[7],
            self.relu
        )

        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu,
        )

        self.conv4 = nn.Sequential(
            self.encoder[17],
            self.relu,
            self.encoder[19],
            self.relu,
            self.encoder[21],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[24],
            self.relu,
            self.encoder[26],
            self.relu,
            self.encoder[28],  # n filters 512
            self.relu,
        )

        if dilation_depth:
            current_channels = 512
            self.center_dilation = CenterDilation(current_channels, dilation_depth, False, dilation_type)
        else:
            self.center_dilation = None

        self.center = UNetVGG16DecoderBlock(
            512, init_features * 8 * 2, init_features * 8, upsampling_type,
        )

        self.dec5 = UNetVGG16DecoderBlock(
            512 + init_features * 8, init_features * 8 * 2, init_features * 8, upsampling_type
        )
        self.dec4 = UNetVGG16DecoderBlock(
            512 + init_features * 8, init_features * 8 * 2, init_features * 8, upsampling_type
        )
        self.dec3 = UNetVGG16DecoderBlock(
            256 + init_features * 8, init_features * 4 * 2, init_features * 2, upsampling_type
        )
        self.dec2 = UNetVGG16DecoderBlock(
            128 + init_features * 2, init_features * 2 * 2, init_features, upsampling_type
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + init_features, init_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(init_features, classes, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center_input = self.pool(conv5)

        if self.center_dilation is not None:
            center_input = self.center_dilation(center_input)

        center = self.center(center_input)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.classifier(dec1)
