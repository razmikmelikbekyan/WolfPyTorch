from enum import Enum
from typing import Set


class HighLevelArchitectureTypes(Enum):
    """Defines the high level Architecture Type."""
    SINGLE_BRANCH = "SINGLE_BRANCH"  # model outputs a single output
    MULTI_BRANCH = "MULTI_BRANCH"  # model has multiple branches and may require multiple loss function and metrics
    TEMPORAL = "TEMPORAL"  # model receives a sequence and returns a sequence

    @classmethod
    def get_members(cls) -> Set[str]:
        return set(x.name for x in cls)


class TaskTypes(Enum):
    """Defines a possible task types."""

    # --------------------------- the main tasks ---------------------------

    # outputs a single number, ordinary regression
    # output size: [B, 1]
    IL_REGRESSION = "IL_REGRESSION", "IMAGE_LEVEL_REGRESSION"

    # outputs a single number for each pixel - regression segmentation
    # output size: [B, 1, H, W]
    PL_REGRESSION = "PL_REGRESSION", "PIXEL_LEVEL_REGRESSION"

    # outputs a single number - the predicted class logits or positive class probability (after sigmoid)
    # output size: [B, 1]
    IL_BINARY_CLASSIFICATION = "IL_BINARY_CLASSIFICATION", "IMAGE_LEVEL_BINARY_CLASSIFICATION"

    # outputs a single number for each pixel - the predicted class logits or positive class probability (after sigmoid)
    # this is the same as binary segmentation
    # output size: [B, 1, H, W]
    PL_BINARY_CLASSIFICATION = "PL_BINARY_CLASSIFICATION", "PIXEL_LEVEL_BINARY_CLASSIFICATION"

    # outputs a vector - the predicted classes logits or classes probabilities (after softmax)
    # output size: [B, N-Classes]
    IL_MULTI_CLASSIFICATION = "IL_MULTI_CLASSIFICATION", "IMAGE_LEVEL_MULTI_CLASSIFICATION"

    # outputs a vector for each pixel - the predicted classes logits or classes probabilities (after softmax)
    # this is the same as multiclass segmentation
    # output size: [B, N-Classes, H, W]
    PL_MULTI_CLASSIFICATION = "PL_MULTI_CLASSIFICATION", "PIXEL_LEVEL_MULTI_CLASSIFICATION"

    # --------------------------- additional regression tasks ---------------------------

    # outputs quantiles, ordinary quantile regression
    # output size: [B, N-Quantiles]
    IL_QUANTILE_REGRESSION = "IL_QUANTILE_REGRESSION", "IMAGE_LEVEL_QUANTILE_REGRESSION"

    # outputs quantiles fro each pixel, ordinary quantile regression for each pixel
    # output size: [B, N-Quantiles, H, W]
    PL_QUANTILE_REGRESSION = "PL_QUANTILE_REGRESSION", "PIXEL_LEVEL_QUANTILE_REGRESSION"

    # outputs pi, mean and sigma, ordinary mixture density regression
    # output size: [B, N-MixtureComponents] (for pi, mean and sigma, if combine will be [B, 3 * N-MixtureComponents])
    IL_MDN_REGRESSION = "IL_MDN_REGRESSION", "IMAGELEVEL_MDN_REGRESSION"

    # outputs pi, mean and sigma for each pixel, ordinary mixture density regression for each pixel
    # output size: [B, N-MixtureComponents, H, W]
    # (for pi, mean and sigma, if combine will be [B, 3 * N-MixtureComponents, H, W])
    PL_MDN_REGRESSION = "PL_MDN_REGRESSION", "PIXEL_LEVEL_MDN_REGRESSION"

    # --------------------------- the multi task ---------------------------
    # the multi task case, it should be suitable only for MultiBranch networks
    MULTI_TASK = "MULTI_TASK"

    @classmethod
    def get_image_level_tasks(cls):
        return frozenset([x for x in cls if x.name.startswith('IL_')])

    @classmethod
    def get_pixel_level_tasks(cls):
        return frozenset([x for x in cls if x.name.startswith('PL_')])

    @classmethod
    def get_members(cls) -> Set[str]:
        return set(x.name for x in cls)
