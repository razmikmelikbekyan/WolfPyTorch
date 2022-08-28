from typing import List

import torch

from ..single_branch.base import SingleBranchEpochMixin


class TemporalEpochMixin(SingleBranchEpochMixin):
    """Special class for temporal imagery data."""
    
    IS_TEMPORAL = True

    @staticmethod
    def adjust_temporal_prediction(data: torch.Tensor,
                                   target: torch.Tensor,
                                   prediction: List[torch.Tensor]) -> torch.Tensor:
        """Adjusts the temporal prediction."""
        if data.ndim == 5:
            # the input must be: B x T x *
            B, T, C, H, W = data.shape
            assert target.shape[:2] == (B, T), target.shape
        elif data.ndim == 3:
            # the input must be: B x T x NFeatures
            B, T, _ = data.shape
            assert target.shape[:2] == (B, T), target.shape
        else:
            raise ValueError(f"Data must be 5 or 3 dimensional, got {data.shape}.")

        # prediction must be list with length of T and each element must be tensor with B x *
        assert isinstance(prediction, list), type(prediction)
        assert len(prediction) == T, len(prediction)
        assert prediction[0].shape[0] == B, prediction[0].shape

        return torch.transpose(torch.stack(prediction, dim=0), 0, 1)  # B x T *

    def register_loss_metrics(self, *args, **kwargs) -> None:
        raise NotImplementedError
