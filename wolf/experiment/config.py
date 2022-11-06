import json
from copy import deepcopy
from typing import Any, Dict

import torch
from easydict import EasyDict

from ..enums import TaskTypes, HighLevelArchitectureTypes
from ..metrics import MULTI_BRANCH_EVALUATOR_CONFIG_KEYS

__all__ = ['BaseExperimentConfig']


class WrongExperimentConfig(Exception):
    """Exception for wrong specified experiment config."""
    pass


class BaseExperimentConfig:
    """Base Class for handling experiment configuration files."""

    __slots__ = (
        '_keeper'
    )

    OBLIGATORY_ARGS = {
        "experiment": frozenset((
            "name",
            "kwargs"
        )),
        "train_data": frozenset((
            "name",
            "kwargs"
        )),
        "valid_data": frozenset((
            "name",
            "kwargs"
        )),
        "model": frozenset((
            "name",
        )),
        "optimizer": frozenset((
            "name",
        )),
        "loss": frozenset((
            "name",
        )),
        "evaluator": frozenset((
            "name",
            "kwargs"
        ))
    }

    OBLIGATORY_KWARGS = {
        "experiment": frozenset((
            "high_level_architecture",
            "task",
            "epochs",
            "device",
            "batch_size",
            "num_workers",
            "verbose_epochs",
            "save_path",
            "early_stopping",
        )),
        "train_data": frozenset((
            "input_file",
        )),
        "valid_data": frozenset((
            "input_file",
        )),
        "model": frozenset((
        )),
        "optimizer": frozenset((
        )),
        "loss": frozenset((
        )),
        "evaluator": frozenset((
        ))
    }

    def __init__(self, config: str or dict):
        """
        Args:
            config: path to JSON file with configurations or the dict with config
        """
        if isinstance(config, str):
            if not config.lower().endswith('.json'):
                raise ValueError(f"Given config file {config} is not JSON file.")

            with open(config) as f:
                config = json.load(f)

        elif isinstance(config, dict):
            config = deepcopy(config)
        else:
            raise TypeError(f"Config must be str or dict, got: {type(config)}")

        self._validate_args(config)
        self._set_enums(config)
        self._validate_kwargs(config)
        self._validate_multi_branch_configuration(config)
        self._keeper = EasyDict(config)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._keeper, item)

    def _adjust_device(self):
        """Sets device to CPU if the GPU is not available"""
        if 'cuda' in self.experiment.kwargs.device:
            if not torch.cuda.is_available():
                self.set_single_config('experiment|device', 'cpu')
            else:
                gpu_ids = list(range(torch.cuda.device_count()))
                self.set_single_config('experiment|device_ids', gpu_ids)

    def _validate_args(self, config: Dict) -> None:
        diff = set(self.OBLIGATORY_ARGS.keys()) - set(config.keys())
        if diff:
            raise WrongExperimentConfig(
                f'Given config must contain all the keys listed here: {list(self.OBLIGATORY_ARGS.keys())}. '
                f'{diff} keys are missing.'
            )

        for k, v in config.items():
            if not isinstance(v, dict):
                raise WrongExperimentConfig(f'Config for {k} must be {dict}, got {type(v)}.')
            if not set(v.keys()).issuperset(self.OBLIGATORY_ARGS[k]):
                raise WrongExperimentConfig(
                    f'Config of {k} must contain the following obligatory keys: {self.OBLIGATORY_ARGS[k]}.'
                )

            if "kwargs" not in v:
                v["kwargs"] = {}

            if not isinstance(v["kwargs"], dict):
                raise WrongExperimentConfig(f'kwargs config for {k} must be {dict}, got {type(v["kwargs"])}.')

    @classmethod
    def _set_enums(cls, config: Dict):
        """Sets enums"""
        enums_mapper = {
            'high_level_architecture': HighLevelArchitectureTypes,
            'task': TaskTypes
        }

        if isinstance(config, dict):
            for k in config.keys():
                value = config[k]
                if k in enums_mapper and isinstance(value, str):
                    enum_klass = enums_mapper[k]
                    try:
                        config[k] = enum_klass[value.upper()]
                    except KeyError:
                        raise WrongExperimentConfig(
                            f'Given {k}={config[k]} is wrong, please select from {enum_klass.get_members()}')

                elif isinstance(value, (list, tuple, set)):
                    for item in value:
                        cls._set_enums(item)
                elif isinstance(value, dict):
                    cls._set_enums(value)
        elif isinstance(config, (list, tuple, set)):
            for item in config:
                cls._set_enums(item)

        return config

    @staticmethod
    def _validate_evaluator_kwargs(evaluator_kwargs):
        """Validates single evaluator kwargs."""
        if not isinstance(evaluator_kwargs['metrics'], list):
            raise WrongExperimentConfig('Evaluator "metrics" must be a list.')

        if not all(isinstance(x, list) and len(x) == 2 for x in evaluator_kwargs['metrics']):
            raise WrongExperimentConfig('Each item in Evaluator "metrics" must be a list with 2 elements.')

        if not all(isinstance(x[0], str) for x in evaluator_kwargs['metrics']):
            raise WrongExperimentConfig('Each first item in Evaluator "metrics" must be a str.')

        if not all(isinstance(x[1], dict) for x in evaluator_kwargs['metrics']):
            raise WrongExperimentConfig('Each second item in Evaluator "metrics" must be a dict.')

    @classmethod
    def _validate_multi_branch_evaluator_kwargs(cls, evaluator_kwargs):
        """Validates multi branch evaluator kwargs."""
        if 'multibranch_evaluator_config' not in evaluator_kwargs:
            raise WrongExperimentConfig('MultiBranchEvaluator kwargs '
                                        'must contain "multibranch_evaluator_config".')

        data = evaluator_kwargs['multibranch_evaluator_config']
        if not isinstance(data, list) and not all(isinstance(item, dict) for item in data):
            raise WrongExperimentConfig('MultiBranchEvaluator must contain a list of dicts for'
                                        'single task evaluators.')

        for item in data:
            obligatory_keys = MULTI_BRANCH_EVALUATOR_CONFIG_KEYS
            if not all(x in item for x in obligatory_keys):
                raise WrongExperimentConfig(f'Each Evaluator must contain these keys: {obligatory_keys}')

            cls._validate_evaluator_kwargs(item['evaluator_kwargs'])

    def _validate_kwargs(self, config: Dict) -> None:
        for k, v in config.items():
            kwargs_keys = set(v.get("kwargs", {}).keys())
            if not kwargs_keys.issuperset(self.OBLIGATORY_KWARGS[k]):
                raise WrongExperimentConfig(
                    f'kwargs config of {k} must contain the following obligatory keys: {self.OBLIGATORY_KWARGS[k]}.'
                )

            if k == 'evaluator':
                if "kwargs" not in v:
                    raise WrongExperimentConfig('Evaluator must contain kwargs.')

                if v['name'] != 'MultiBranchEvaluator':
                    self._validate_evaluator_kwargs(v['kwargs'])
                else:
                    self._validate_multi_branch_evaluator_kwargs(v['kwargs'])

        self._validate_train_valid_data_kwargs(config)

    @staticmethod
    def _validate_train_valid_data_kwargs(config: Dict):
        train_data_config = config['train_data']['kwargs']
        valid_data_config = config['valid_data']['kwargs']

        for name in ('to_use_bands', 'normalize', 'float32_factor'):
            train_name_config = train_data_config.get(name)
            valid_name_config = valid_data_config.get(name)

            if train_data_config is None and valid_data_config is None:
                pass
            else:
                if not train_name_config == valid_name_config:
                    raise ValueError(f"The '{name}' config for train data and valid data must be equal, "
                                     f"got: {train_name_config}, {valid_name_config}")

    @classmethod
    def _validate_multi_branch_configuration(cls, config: Dict):
        """In case of multi branch validates config."""
        high_level_architecture = config['experiment']['kwargs']['high_level_architecture']
        if high_level_architecture == HighLevelArchitectureTypes.MULTI_BRANCH:
            if config['experiment']['kwargs']['task'] != TaskTypes.MULTI_TASK:
                raise WrongExperimentConfig("In case of MULTI_BRANCH architecture the task must be MULTI_TASK")

            if config['loss']['name'] != 'MultiBranchLoss':
                raise WrongExperimentConfig("In case of MULTI_BRANCH architecture the loss must be MultiBranchLoss")

            if config['evaluator']['name'] != 'MultiBranchEvaluator':
                raise WrongExperimentConfig("In case of MULTI_BRANCH architecture the evaluator"
                                            " must be MultiBranchEvaluator")

            loss_tasks = [item['task'] for item in config['loss']['kwargs']['multibranch_loss_config']]
            evaluator_tasks = [item['task'] for item in config['evaluator']['kwargs']['multibranch_evaluator_config']]
            if not all(x == y for x, y in zip(loss_tasks, evaluator_tasks)) or len(loss_tasks) != len(evaluator_tasks):
                raise WrongExperimentConfig("In case of MULTI_BRANCH architecture the evaluator and loss tasks"
                                            " must be the same.")

            loss_branches = [item['branch'] for item in config['loss']['kwargs']['multibranch_loss_config']]
            evaluator_branches = [
                item['branch'] for item in config['evaluator']['kwargs']['multibranch_evaluator_config']
            ]
            if (not all(x == y for x, y in zip(loss_branches, evaluator_branches))
                    or len(loss_branches) != len(evaluator_branches)):
                raise WrongExperimentConfig("In case of MULTI_BRANCH architecture the evaluator and loss branches"
                                            " must be the same.")

    def set_single_config(self, key: str, value: Any):
        """Attribute setting implementation."""
        keys = [x.strip() for x in key.split('|')]

        if len(keys) == 2:
            [main_config_name, inner_config_name] = keys
            nested_config_name = None
        elif len(keys) == 3:
            [main_config_name, inner_config_name, nested_config_name] = keys
        else:
            raise WrongExperimentConfig(f'Config setting is supported only with depth=3, got {len(keys)}')

        if main_config_name not in self.OBLIGATORY_ARGS:
            raise WrongExperimentConfig(
                f'Given config name {main_config_name} not in possible configs. '
                f'Please see the list of possible inputs: {list(self.OBLIGATORY_KWARGS.keys())}'
            )

        if inner_config_name == 'name':
            self._keeper[main_config_name][inner_config_name] = value
        else:
            if not nested_config_name:
                self._keeper[main_config_name]['kwargs'][inner_config_name] = value
            else:
                self._keeper[main_config_name]['kwargs'][inner_config_name][nested_config_name] = value

    @property
    def all_configs(self) -> Dict:
        """Returns all configs."""
        return deepcopy(self._keeper)

    def get_hyper_params_to_log(self) -> Dict:
        """Returns the hyper-parameters of the experiment to log."""
        return {
            "experiment_name": self.experiment.name,
            "high_level_architecture": self.experiment.kwargs['high_level_architecture'].name,
            "task": self.experiment.kwargs['task'].name,
            "epochs": self.experiment.kwargs['epochs'],
            "batch_size": self.experiment.kwargs['batch_size'],
            "train_steps_per_epoch": self.experiment.kwargs.get('train_steps_per_epoch', None),
            "valid_steps_per_epoch": self.experiment.kwargs.get('valid_steps_per_epoch', None),
            "early_stopping_patience": self.experiment.kwargs.get("early_stopping", {}).get("patience", None),
            "model": self.model.name,
            "loss": self.loss.name,
            "optimizer": self.optimizer.name,
            "lr": self.optimizer.kwargs.get("lr", None),
            "scheduler": self.optimizer.get("lr_scheduler", None),
        }
