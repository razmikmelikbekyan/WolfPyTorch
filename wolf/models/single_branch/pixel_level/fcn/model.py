import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101

from ....helpers.weights.adjust_first_conv import patch_first_conv


class FCNResnet(nn.Module):
    """
    FCN with resnet backbone.
    Adapted from here: https://pytorch.org/vision/0.8/models.html#semantic-segmentation
    """

    def __init__(self, in_channels: int, classes: int, pretrained: bool, backbone: str = 'resnet50'):
        super(FCNResnet, self).__init__()
        if backbone == 'resnet50':
            self._model = fcn_resnet50(pretrained=pretrained, progress=True)
        elif backbone == 'resnet101':
            self._model = fcn_resnet101(pretrained=pretrained, progress=True)
        else:
            raise NotImplementedError

        self._model.aux_classifier = None  # we do not need this
        patch_first_conv(self._model, in_channels)
        self._model.classifier[4] = nn.Conv2d(512, classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self._model.forward(x)['out']
