import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def read_file(file_path: str or Path) -> Any:
    """Reads and returns the content of file."""
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.json':
        with open(file_path) as f:
            data = json.load(f)
    elif file_path.suffix.lower() == '.csv':
        data = pd.read_csv(file_path)
    elif file_path.suffix.lower() == '.pkl':
        data = pd.read_pickle(file_path)
    elif file_path.suffix.lower() == '.yaml':
        with open(file_path, 'r') as f:
            loader = yaml.SafeLoader
            loader.add_implicit_resolver(
                u'tag:yaml.org,2002:float',
                re.compile(u'''^(?:
                       [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                       |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                       |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                       |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                       |[-+]?\\.(?:inf|Inf|INF)
                       |\\.(?:nan|NaN|NAN))$''', re.X),
                list(u'-+0123456789.'))
            data = yaml.load(f, Loader=loader)
    else:
        raise ValueError("Given file must be JSON, CSV, PKL or YAML")
    return data


def save_dict_of_objects(dict_data: Dict):
    """Saves the given object in the given file paths."""
    for file_path, data in dict_data.items():
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        save_file_extension = file_path.suffix.lower()
        if save_file_extension == '.json':
            with open(file_path, 'w+') as f:
                json.dump(data, f)
        elif save_file_extension == '.csv':
            data.to_csv(file_path)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


