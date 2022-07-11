# code is adapted from here: https://github.com/DeepVoltaire/Fast-SCNN

import torch
from torch import nn
from torch.nn import functional as F

from ....helpers.layers.cnn import conv3x3, conv1x1, ConvBNActivation


class FastSCNN(nn.Module):
    """Fast_SCNN is a fast convolutional neural network for image semantic segmentation"""

    def __init__(self, in_channels: int, classes: int):
        """
        Args:
            in_channels: number of channels in the input image
            classes: number of classes
        """
        super().__init__()

        self.learning_to_downsample = LearningToDownsample(in_channels)
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusionModule(scale_factor=4)
        self.classifier = Classifier(classes, scale_factor=8)

    def forward(self, x):
        shared = self.learning_to_downsample(x)
        x = self.global_feature_extractor(shared)
        x = self.feature_fusion(shared, x)
        x = self.classifier(x)
        return x


class LearningToDownsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv1 = ConvBNActivation(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2,
                                      padding=1, dilation=1, batch_norm=True, dropout_rate=0.0)
        self.sconv1 = nn.Sequential(
            conv3x3(32, 32, stride=2, groups=32, dilation=1),
            nn.BatchNorm2d(32),
            conv1x1(32, 48),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.sconv2 = nn.Sequential(
            conv3x3(48, 48, stride=2, groups=48, dilation=1),
            nn.BatchNorm2d(48),
            conv1x1(48, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.sconv1(x)
        x = self.sconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_block = nn.Sequential(
            InvertedResidual(64, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
        )
        self.second_block = nn.Sequential(
            InvertedResidual(64, 96, 2, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
        )
        self.third_block = nn.Sequential(
            InvertedResidual(96, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6),
            InvertedResidual(128, 128, 1, 6)
        )
        self.ppm = PSPModule(128, 128)

    def forward(self, x):
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)
        x = self.ppm(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                conv3x3(hidden_dim, hidden_dim, stride=stride, dilation=1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv1x1(hidden_dim, oup),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                conv1x1(inp, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv3x3(hidden_dim, hidden_dim, stride=stride, dilation=1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv1x1(hidden_dim, oup),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()

        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = conv1x1(features * (len(sizes) + 1), out_features, bias=True)
        self.relu = nn.ReLU()

    @staticmethod
    def _make_stage(features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = conv1x1(features, features)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages
        ]
        priors += [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class FeatureFusionModule(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor
        self.sconv1 = ConvBNActivation(
            in_channels=128, out_channels=128, kernel_size=3, stride=1,
            padding=scale_factor, dilation=scale_factor, batch_norm=True, dropout_rate=0.0, groups=128
        )
        self.conv_low_res = conv1x1(128, 128, bias=True)
        self.conv_high_res = conv1x1(64, 128, bias=True)
        self.relu = nn.ReLU()

    def forward(self, high_res_input, low_res_input):
        low_res_input = F.interpolate(
            input=low_res_input, scale_factor=self.scale_factor, mode='bilinear', align_corners=True
        )
        low_res_input = self.sconv1(low_res_input)
        low_res_input = self.conv_low_res(low_res_input)
        high_res_input = self.conv_high_res(high_res_input)
        x = torch.add(high_res_input, low_res_input)
        return self.relu(x)


class Classifier(nn.Module):
    def __init__(self, num_classes, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor
        self.sconv1 = ConvBNActivation(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                       padding=1, dilation=1, batch_norm=True, dropout_rate=0.0, groups=128)
        self.sconv2 = ConvBNActivation(in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                       padding=1, dilation=1, batch_norm=True, dropout_rate=0.0, groups=128)
        self.conv = conv1x1(128, num_classes, bias=True)

    def forward(self, x):
        x = self.sconv1(x)
        x = self.sconv1(x)
        x = self.conv(x)
        x = F.interpolate(input=x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return x
