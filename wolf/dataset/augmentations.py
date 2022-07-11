from typing import Dict, Tuple
from albumentations import (
    BasicTransform,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    VerticalFlip,
    HorizontalFlip,
    Transpose,
    Compose,
    RandomRotate90,
    Rotate,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomBrightnessContrast,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    ShiftScaleRotate,
    Resize,
    CoarseDropout,
    Cutout,
    RandomShadow
)

__all__ = ['AUGMENTATIONS', 'get_augmentation', 'initialize_augmentations']

AUGMENTATIONS = {
    'CenterCrop': CenterCrop,  # Crop the central part of the input.
    'RandomCrop': RandomCrop,  # Pad side of the image / max if side is less than desired number.
    'PadIfNeeded': PadIfNeeded,  # Pad side of the image / max if side is less than desired number.
    'Rotate': Rotate,  # Rotate the input by an angle selected randomly from the uniform distribution.
    'ShiftScaleRotate': ShiftScaleRotate,  # Randomly apply affine transforms: translate, scale and rotate the input.
    'Resize': Resize,

    # Non destructive transformations
    'VerticalFlip': VerticalFlip,  # Flip the input vertically around the x-axis.
    'HorizontalFlip': HorizontalFlip,  # Flip the input horizontally around the y-axis.
    'RandomRotate90': RandomRotate90,  # Randomly rotate the input by 90 degrees zero or more times.
    'Transpose': Transpose,  # Transpose the input by swapping rows and columns.

    # Non-rigid transformations
    'ElasticTransform': ElasticTransform,
    'GridDistortion': GridDistortion,
    'OpticalDistortion': OpticalDistortion,

    # Son-spatial transformations
    'RandomBrightnessContrast': RandomBrightnessContrast,
    'RandomBrightness': RandomBrightness,
    'RandomContrast': RandomContrast,
    'RandomGamma': RandomGamma,

    # other
    'CoarseDropout': CoarseDropout,  # CoarseDropout of the rectangular regions in the image
    'Cutout': Cutout,  # CoarseDropout of the square regions in the image
    'RandomShadow': RandomShadow
}


def get_augmentation(name: str) -> BasicTransform:
    """Returns the augmentation class based on its name."""
    try:
        return AUGMENTATIONS[name]
    except KeyError:
        raise ValueError(
            f'Given augmentation {name} is not supported.'
            f'Please select from: {" | ".join(AUGMENTATIONS.keys())}.'
        )


def initialize_augmentations(*names_kwargs: Tuple[str, Dict], additional_targets: Dict = None) -> Compose:
    """Compose transforms."""
    if not names_kwargs:
        return
    return Compose([get_augmentation(name)(**kwargs) for name, kwargs in names_kwargs],
                   additional_targets=additional_targets)
