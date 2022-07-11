from abc import abstractmethod, ABC
from pathlib import Path
from typing import Dict, List, Union, Any

import numpy as np
import pandas as pd
import torch
from albumentations import Compose
from torch.utils.data import Dataset

from yield_forecasting.utils.logger import logger
from .augmentations import initialize_augmentations


class BaseDataset(Dataset, ABC):
    """Base class for all datasets which are using images as an input."""

    @property
    @abstractmethod
    def INPUT_BANDS(self):
        """Defines the main input bands.

        Examples:
            INPUT_BANDS = ['red', 'green', 'blue', 'nir']
        """

    def __init__(self,
                 input_file: str or pd.DataFrame,
                 to_use_bands: List[str],
                 input_depth: str,
                 debug_band: str or List[str] = None,
                 shuffle: bool = False,
                 augmentations: Compose or List[Dict] = None,
                 augmentations_additional_targets: Dict = None,
                 normalize: Dict = None,
                 float32_factor: str or int or Dict = 'default',
                 process_initial_df: bool = True,
                 sample_size: int = None,
                 random_seed: int = 42,
                 steps_per_epoch: int = None,
                 ):
        """

        Args:
            input_file: the file path which contains images paths, for each of bands it should provide image path
                        or the dataframe
            to_use_bands: the list of BANDs to use during dl_experiments, this is used in the case when some new bands can be
                          created based on input bands
            input_depth: the input depth of the input images
            debug_band: the name of the band which will be used for network debugging,
                        in most cases it is going to be RGB
            shuffle: if True, shuffles data
            augmentations: the image augmentations to be used
            augmentations_additional_targets: the additional targets to use by augmentations,
                                              check the link below for the format:
                                              https://albumentations.ai/docs/examples/example_multi_target/
            normalize: the dictionary with the configuration to make normalization,
                       {"method": "min_max", "kwargs": {}} or
                       {"method": "mean_std", "kwargs": {"means_stds": {"RED": (0.5, 0.1)}}
            float32_factor: the factor for converting images to float32 before giving to neural network
            process_initial_df: if True processes initial data
            sample_size: the number of samples to select, if not given will select all the samples
            random_seed: random seed
            steps_per_epoch: the number of steps to perform per epoch, if not given will use all steps
        """
        if debug_band and isinstance(debug_band, str) and debug_band not in to_use_bands:
            raise ValueError(f"debug band must be from {to_use_bands}, got: {debug_band}.")
        if debug_band and isinstance(debug_band, list) and not set(to_use_bands).issuperset(set(debug_band)):
            raise ValueError(f"debug band must be from {to_use_bands}, got: {debug_band}.")

        self._df = self.read_input_file(input_file).reset_index(drop=True)
        logger.info(f"Initial Data contains contains N={len(self._df)} samples.")
        self._process_initial_df = process_initial_df

        self._to_use_bands = to_use_bands
        self._input_depth = input_depth
        self._debug_band = debug_band

        self._normalize = normalize
        self._float32_factor = float32_factor

        augmentations_additional_targets = augmentations_additional_targets or dict()
        augmentations_additional_targets.update({"debug": "image"})
        self._augmentations = self.get_augmentations(augmentations, additional_targets=augmentations_additional_targets)

        self._random_seed = random_seed
        self._shuffle = shuffle
        self._sample_size = sample_size
        self._steps_per_epoch = steps_per_epoch

        self._df = self.process_df()
        if sample_size:
            sample_size = min(len(self._df), sample_size)
            self._df = self._df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
            logger.info(f'After sampling data contains N={len(self._df)} samples')

        self._sample_providers = self.get_sample_providers()

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def to_use_bands(self) -> List[str]:
        return self._to_use_bands.copy()

    @property
    def debug_band(self) -> str or List[str]:
        return self._debug_band

    @property
    def steps_per_epoch(self) -> int or None:
        return self._steps_per_epoch

    def get_sample_weights(self) -> int or None:
        raise NotImplementedError

    @staticmethod
    def get_augmentations(augmentations: Compose or List[Dict], additional_targets: Dict = None) -> Compose or None:
        """Returns augmentations as Compose object."""

        if isinstance(augmentations, Compose):
            if additional_targets:
                augmentations.additional_targets = additional_targets
                augmentations.add_targets(additional_targets=additional_targets)
            return augmentations
        elif isinstance(augmentations, list):
            augmentations = initialize_augmentations(
                *[(x['name'], x['kwargs']) for x in augmentations],
                additional_targets=additional_targets
            )
            return augmentations
        elif augmentations is None:
            return
        else:
            raise ValueError(f"augmentations must be list of dicts or Compose, got: {type(augmentations)}")

    @property
    def augmentations(self) -> Compose:
        return self._augmentations

    def apply_augmentation(self, sample: Dict[str, np.ndarray]):
        """Override this method to apply augmentation of your dataset."""
        raise NotImplementedError

    def process_df(self) -> pd.DataFrame:
        """
        Override this method to process the given data of your dataset class.
        An each row of the resulting dataframe must be one dl_experiments sample.
        """
        raise NotImplementedError

    def get_sample_providers(self) -> List[Dict]:
        """Override this method to return ImageProvider objects and any other info used during dl_experiments."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._df)

    @staticmethod
    def dict_to_tensor(sample: Dict[str, Union[np.ndarray, Any]]) -> Dict[str, Union[torch.Tensor, float]]:
        """Converts dict of items to dict of tensors."""
        output = {}
        for k, v in sample.items():
            if not isinstance(v, np.ndarray):
                output[k] = v
            else:
                out_sample = np.nan_to_num(v, nan=0, posinf=0, neginf=0)
                if k == 'input_image' or k == 'debug_band' or k == 'mask':
                    out_sample = np.transpose(out_sample, axes=(2, 0, 1))
                output[k] = torch.from_numpy(out_sample)

        return output

    @staticmethod
    def read_input_file(input_file: str or pd.DataFrame) -> pd.DataFrame:
        """Reads the input file and returns pandas DataFrame."""
        if isinstance(input_file, pd.DataFrame):
            return input_file.copy()
        else:
            input_file = Path(input_file)

            if not input_file.exists():
                raise FileNotFoundError(f"Given '{input_file}' input_images_info_file does not exist.")

            mapper = {
                '.json': pd.read_json,
                '.csv': pd.read_csv,
                '.pkl': pd.read_pickle
            }
            try:
                return mapper[input_file.suffix.lower()](input_file)
            except KeyError:
                raise NotImplementedError(
                    f"For given suffix '{input_file.suffix.lower()}' the data reader is not implemented."
                )
