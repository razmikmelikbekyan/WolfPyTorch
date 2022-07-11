import torch.nn as nn


class Loss(nn.Module):
    """Base class for all losses."""

    def __init__(self, name: str = None):
        super().__init__()
        self._name = name

    @property
    def name(self):
        return self._name if self._name is not None else self.__class__.__name__

    def __add__(self, other):
        """Defines addition of 2 losses_."""
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)


class SumOfLosses(Loss):
    """Helper class for summing 2 losses_."""

    def __init__(self, l1: Loss, l2: Loss):
        super().__init__(name=f'{l1.name} + {l2.name}')
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)
