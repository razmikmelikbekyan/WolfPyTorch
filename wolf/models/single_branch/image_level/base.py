import timm
import torch.nn as nn

from ...helpers.weights.adjust_first_conv import patch_first_conv


class ImageLevelModel(nn.Module):
    """A simple model that is based on timm package (https://github.com/rwightman/pytorch-image-models).
    It is basically extract the features and uses linear layers at the end.
    In case of classification models it will output the raw logits - un-normalized probabilities

    It supports the following task types:
        - image level regression
        - image level binary classification
        - image level multi classification
        - image level quantile regression
    """

    def __init__(self,
                 architecture: str,
                 in_channels: int = 3,
                 dropout_rate: float = None,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = True,
                 out_channels: int = 1,
                 ):
        """

        Args:
            architecture: the encoder architecture name, like vgg16, resnet34 and so on
            in_channels: number of input channels
            dropout_rate: the optional dropout rate
            pretrained: if True will use the imagenet pretrained weights
            freeze_feature_extractor: if true will freeze the feature extractor and update only
                                     the last linear head (Classification head)
            out_channels: the number of output channels:
                            - in case of regression or binary classification it must be 1,
                            - in case of multi classification it should be equal to number of classes,
                            - in case of quantile regression it will be equal to number of quantiles
                            - in case of mixture density regression it will be equal 3 * mixture components

        """
        super().__init__()

        if architecture not in timm.list_models(pretrained=pretrained):
            raise ValueError(f"Given name={architecture} is not supported.")

        self.timm_model = timm.create_model(
            architecture, pretrained=pretrained,
            drop_rate=dropout_rate, num_classes=out_channels
        )

        if in_channels != 3:
            patch_first_conv(self.timm_model, in_channels)

        if freeze_feature_extractor:
            self.freeze_feature_extractor()
            self.timm_model.reset_classifier(num_classes=out_channels)

    def freeze_feature_extractor(self):
        """Freezes model parameters except the classification head."""
        for param in self.timm_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.timm_model(x)

    @property
    def n_features(self) -> int:
        return self.timm_model.num_features
