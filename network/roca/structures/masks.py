from typing import List

import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import BitMasks


class Masks(BitMasks):
    def crop_and_resize_with_grid(
        self,
        grid: torch.Tensor,
        crop_size: int,  # consistency of API
    ) -> torch.Tensor:
        grid = grid.permute(0, 2, 3, 1)
        tensor = self.tensor.unsqueeze(1)
        crops = F.grid_sample(
            tensor.to(grid.dtype), grid, 'nearest', align_corners=False
        )
        return crops

    def to(self, *args, **kwargs) -> "Masks":
        return self.__class__(tensor=self.tensor.to(*args, **kwargs))

    def __getitem__(self, index) -> "Masks":
        masks = super().__getitem__(index)
        return self.__class__(tensor=masks.tensor)

    @classmethod
    def cat(cls, masks: List["Masks"]) -> "Masks":
        tensor = cat([mask.tensor for mask in masks], dim=0)
        return cls(tensor=tensor)
