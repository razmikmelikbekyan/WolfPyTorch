from collections import defaultdict
from typing import Tuple, Dict

from wolf.experiment import BaseExperimentConfigParser, BaseExperiment
from dataset import MNISTDataset
from argparse import ArgumentParser


class MNISTExperimentConfigParser(BaseExperimentConfigParser):
    """MNIST experiments base config parser."""

    def get_datasets(self) -> Tuple[MNISTDataset, MNISTDataset]:
        return (
            MNISTDataset(**self.config.train_data.kwargs),
            MNISTDataset(**self.config.valid_data.kwargs),
        )


class MNISTExperiment(BaseExperiment):
    """MNIST Experiment."""

    CONFIG_PARSER_KLASS = MNISTExperimentConfigParser

    def generate_inference_config(self) -> Dict:
        """Generates inference config."""
        if self.logger.cleaned_up:
            return {}
        else:
            config = defaultdict(dict)
            config['model'] = self.config.model.copy()
            config['model']['high_level_architecture'] = self.config.experiment.kwargs.high_level_architecture.name
            config['model']['task'] = self.config.experiment.kwargs.task.name
            config['model']['kwargs']['path'] = str(self.logger.model_weights_path)
            config['data']['float32_factor'] = self.config.valid_data.kwargs.get('float32_factor')
            config['data']['normalize'] = self.config.valid_data.kwargs.get('normalize')
            config['data']['batch_size'] = self.config.experiment.kwargs.get('batch_size', 1)
            return config


if __name__ == '__main__':
    parser = ArgumentParser(description="MNSIT Training script.")
    parser.add_argument('--config_json', type=str, required=True, help="The path of the training config file.")

    args = parser.parse_args()
    experiment = MNISTExperiment(args.config_json)
    experiment.run()

