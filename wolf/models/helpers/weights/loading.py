import os
import sys
from urllib.request import urlretrieve

import torch
import torch.nn as nn

from yield_forecasting.utils.logger import logger


def load_model_from_path(model_instance: nn.Module, model_path: str or None) -> torch.nn.Module:
    """Returns the loss class based on its name."""
    if model_path is not None:
        if issubclass(type(model_instance), nn.Module):
            checkpoint = torch.load(os.path.expanduser(model_path), map_location='cpu')

            try:
                model_weights = checkpoint["model_state_dict"]
            except KeyError:
                model_weights = checkpoint

            # special case for TileLevelModel model
            if all(k.startswith('timm_model.') for k in model_instance.state_dict().keys()):
                model_weights = {
                    (f"timm_model.{k}" if not k.startswith('timm_model.') else k): v
                    for k, v in model_weights.items()
                }

            # special case for PixelLevelSMPModel model
            elif all(k.startswith('smp_model.') for k in model_instance.state_dict().keys()):
                model_weights = {
                    (f"smp_model.{k}" if not k.startswith('smp_model.') else k): v
                    for k, v in model_weights.items()
                }

            try:
                model_instance.load_state_dict(model_weights)
            except RuntimeError:
                model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
                model_instance.load_state_dict(model_weights)
            logger.info(f"Model loaded from model_path={model_path}.")

    return model_instance


def load_url(url: str, model_dir: str = './pretrained'):
    """Downloads model weights"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location='cuda' if torch.cuda.is_available() else 'cpu')
