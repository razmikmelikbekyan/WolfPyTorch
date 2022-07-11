import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,
            bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ConvBNActivation(nn.Module):
    """CONV -> BatchNorm (optional) -> ReLu -> Dropout (optional)"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 dilation: int,
                 batch_norm: bool,
                 dropout_rate: float,
                 groups: int = 1,
                 bias: bool = False
                 ):
        super(ConvBNActivation, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                              groups=groups, bias=bias)
        self.activation = nn.ReLU()

        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, num_channels=16):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        return out


class CenterDilation(nn.Module):
    """Center dilation block."""

    DILATION_TYPES = (
        'cascade',
        'parallel'
    )

    def __init__(self, in_channels: int, dilation_depth: int, batch_norm: bool, dilation_type: str, bias: bool = False):
        super(CenterDilation, self).__init__()

        self.dilation_depth = dilation_depth
        self.dilation_type = dilation_type

        if dilation_type not in self.DILATION_TYPES:
            raise ValueError(
                f'bottleneck_type must be one of the followings {self.DILATION_TYPES} got: "{dilation_type}" .'
            )

        self.dilation_layers = nn.ModuleList()
        for i in range(dilation_depth):
            self.dilation_layers.append(
                ConvBNActivation(
                    in_channels, in_channels,
                    kernel_size=3, dilation=2 ** i, padding=2 ** i, stride=1,
                    batch_norm=batch_norm, dropout_rate=0, bias=bias
                )
            )

    def forward(self, x: torch.Tensor):
        outputs = []
        for dilation_layer in self.dilation_layers:
            if self.dilation_type == 'cascade':
                x = dilation_layer(x)
                outputs.append(x.unsqueeze(-1))
            else:
                outputs.append(dilation_layer(x).unsqueeze(-1))

        x = torch.cat(outputs, dim=-1)
        x = torch.sum(x, dim=-1)
        return x
