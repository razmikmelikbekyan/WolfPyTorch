from typing import List

import torch
import torch.nn as nn

from ....logger import logger


def model_to_device(model: nn.Module, device: str = "cuda", multi_gpu: bool = True, device_ids: List[int] = None):
    """Transfers model to cuda. If multi_gpu is True will use the nn.DataParallel."""
    cuda_is_available = torch.cuda.is_available()

    if device == 'cpu':
        model = model.to("cpu")
        logger.info("Model will use CPU.")
        return model

    if not cuda_is_available:
        model = model.to("cpu")
        logger.warning("No cuda is available, model will use CPU.")
        return model

    if not multi_gpu:
        assert device.startswith('cuda'), device
        model = model.to(device)
        logger.info(f"Model is loaded with device={device}")
    else:
        available_devices = torch.cuda.device_count()
        if device_ids is not None:
            device_ids = sorted(device_ids)[:available_devices]
        else:
            device_ids = list(range(available_devices))

        if not device_ids:
            raise ValueError("cuda is available but no-device ids are identified.")
        elif len(device_ids) == 1:
            # single device
            device = f'cuda:{device_ids[0]}'
            model = model.to(device)
            logger.info(f"Model is loaded with device={device}")
        else:
            # multiple devices
            model = nn.DataParallel(model, device_ids=device_ids)
            model.to(device)
            logger.info(f"Model is loaded with multi GPUs using DataParallel - device_ids={device_ids}.")

    return model
