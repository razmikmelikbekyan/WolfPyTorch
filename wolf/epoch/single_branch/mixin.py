from typing import Set, Any

import torch

from .base import SingleBranchEpochMixin


class SimpleSingleChannelEpochMixin(SingleBranchEpochMixin):
    """This is a Special Mixin for the models that are outputting single channel output,
    which means that the output is either as single number or single channel matrix.

    It can be used for the following tasks:
        - tile level regression
        - tile level binary classification
        - tile level multi classification
        - pixel level regression
        - pixel level binary classification
    """

    def register_sample(self,
                        batch_number: int,
                        debug_image: torch.Tensor,
                        y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        saving_batches: Set[int] = None,
                        **kwargs: Any):
        """Registers samples prediction in the images keeper."""
        # TODO: add assertion about y_true and y_pred shapes
        if saving_batches and batch_number in saving_batches:

            self._results_keeper['images'][batch_number]['y_true'] = y_true.numpy()
            self._results_keeper['images'][batch_number]['y_pred'] = y_pred.numpy()

            try:
                self._results_keeper['images'][batch_number]['debug_image'] = debug_image.detach().cpu().numpy()
            except AttributeError:
                self._results_keeper['images'][batch_number]['debug_image'] = debug_image


class PLMultiClassificationEpochMixin(SingleBranchEpochMixin):
    """This is special class for registering results for pixel level multi classification."""

    def register_sample(self,
                        batch_number: int,
                        debug_image: torch.Tensor,
                        y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        saving_batches: Set[int] = None,
                        **kwargs: Any):
        raise NotImplementedError
