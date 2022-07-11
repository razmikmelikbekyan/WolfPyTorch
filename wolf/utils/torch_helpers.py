from typing import Union, List, Dict, Tuple

import torch


def to_device(obj: Union[torch.Tensor, Dict, List, Tuple], device: str, detach: bool = False):
    """Converts given object to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(device=device) if detach else obj.to(device=device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device, detach=detach) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device, detach=detach) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([to_device(v, device, detach=detach) for v in obj])
    else:
        raise TypeError(f"Invalid type for object, it must be tensor, dict, list or tuple, got: {type(obj)}")


def detach_object(obj: Union[torch.Tensor, Dict, List, Tuple]):
    """Converts detaches given objects."""
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_object(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([detach_object(v) for v in obj])
    else:
        raise TypeError(f"Invalid type for object, it must be tensor, dict, list or tuple, got: {type(obj)}")
