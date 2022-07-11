from typing import List

import torch
import torch.nn as nn


# TODO: add test
class WeightsRegularization:
    """Special loss for regularizing the network weights."""

    POSSIBLE_NAMES = ('L1', 'L2')

    def __init__(self, name: str, weight_names: List[str] or str, factor: float = 0.0005):
        """
        Args:
            name: regularization type L1 or L2
            weight_names: the network weight names for applying regularization
            factor: the factor of the regularization
        """
        if name not in self.POSSIBLE_NAMES:
            raise ValueError(f"Given regularization name={name} is not correct, select from '{self.POSSIBLE_NAMES}'.")

        self.p = 1 if name == 'L1' else 2
        self.factor = factor
        self.weight_names = set(weight_names)

    def forward(self, model: nn.Module):
        reg_loss = 0.
        if 'all' == self.weight_names:
            for param in model.parameters():
                reg_loss += torch.norm(param.view(-1), p=self.p)
        else:
            selected_params = {k: v for k, v in model.named_parameters() if k in self.weight_names}
            diff = self.weight_names.difference(set(selected_params.keys()))
            if diff:
                raise ValueError(f'Given weight names {list(diff)} does not exist in model parameters.')
            else:
                for param in selected_params.values():
                    reg_loss += torch.norm(param.view(-1), p=self.p)

        return reg_loss * self.factor
