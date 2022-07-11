import torch.nn as nn


def patch_first_conv(model: nn.Module, in_channels: int):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaming normal initialization
    """
    # get first conv
    for x in model.modules():
        if isinstance(x, nn.Conv2d):
            first_conv = x
            break
    else:
        raise ValueError('Model does not contain nn.Conv2d layer.')

    # change input channels for first conv
    first_conv.in_channels = in_channels
    weight = first_conv.weight.detach()

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        weight = weight.repeat(1, in_channels // 3 + 1, 1, 1)[:, :in_channels]

    first_conv.weight = nn.Parameter(weight)


def patch_first_3dconv(model: nn.Module, in_channels: int):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaming normal initialization
    """
    # get first conv
    for x in model.modules():
        if isinstance(x, nn.Conv3d):
            first_conv = x
            break
    else:
        raise ValueError('Model does not contain nn.Conv2d layer.')

    # change input channels for first conv
    first_conv.in_channels = in_channels
    weight = first_conv.weight.detach()

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        weight = weight.repeat(1, in_channels // 3 + 1, 1, 1, 1)[:, :in_channels]

    first_conv.weight = nn.Parameter(weight)
