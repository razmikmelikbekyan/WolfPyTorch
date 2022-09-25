from typing import Dict

import cv2
import numpy as np

from wolf.dataset import WolfDataset


class MNISTDataset(WolfDataset):

    def apply_augmentation(self, sample: Dict[str, np.ndarray]):
        if not self.augmentations:
            return sample
        else:
            augmented = self.augmentations(
                image=sample['input_image'],
            )
            return {
                'input_image': augmented['image'],
                'label': sample['label']
            }

    def __getitem__(self, item: int):
        sample_info = self.df.iloc[item]
        input_image = cv2.imread(sample_info['image_path'], -1)
        input_image = np.expand_dims(input_image, axis=-1)
        label = sample_info['label']
        sample = {'input_image': input_image, 'label': label}
        sample = self.apply_augmentation(sample)
        return self.dict_to_tensor(sample)


