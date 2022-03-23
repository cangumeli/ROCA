from typing import List

import torch

import detectron2.layers as L


class Intrinsics(object):
    def __init__(self, tensor: torch.Tensor):
        assert tensor.ndim in (2, 3)
        assert tensor.shape[-2:] == (3, 3), 'Provide the full 3x3 matrix'
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        self.tensor = tensor

    def __len__(self):
        return self.tensor.size(0)

    @property
    def device(self):
        return self.tensor.device

    def to(self, device: str) -> "Intrinsics":
        return type(self)(tensor=self.tensor.to(device=device))

    def clone(self) -> "Intrinsics":
        obj = type(self)(self.tensor.clone())
        return obj

    def __getitem__(self, idx) -> "Intrinsics":
        obj = type(self)(self.tensor[idx])
        return obj

    def __iter__(self):
        yield from self.tensor

    def __repr__(self):
        return "Intrinsics(tensor={})".format(str(self.tensor))

    @staticmethod
    def cat(intr_list: List["Intrinsics"]) -> "Intrinsics":
        return Intrinsics(tensor=L.cat([
            im.tensor for im in intr_list
        ], dim=0))
