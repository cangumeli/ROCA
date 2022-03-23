from typing import List

import numpy as np
import quaternion
import torch

from detectron2.layers import cat
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    standardize_quaternion,
)


class AlignmentBase(object):
    def __init__(self, tensor: torch.Tensor, ndim: int = 3):
        '''
        Args:
            tensor: Nxndim matrix
        '''
        # See Boxes in detectron2.structures

        device = tensor.device \
            if isinstance(tensor, torch.Tensor) else torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, ndim)).to(
                dtype=torch.float32,
                device=device
            )
        assert tensor.dim() == 2 and tensor.size(-1) == ndim, tensor.size()

        self.tensor = tensor.contiguous()

    @classmethod
    def new_empty(cls, batch_size=1, device=None) -> 'AlignmentBase':
        tensor = torch.zeros(batch_size, cls.ndim, device=device)
        if hasattr(cls, 'identity'):
            tensor[:, :] = torch.tensor(cls.identity)
        return cls(tensor=tensor)

    def empty(self) -> torch.Tensor:
        empty = torch.zeros_like(self.tensor)
        cls = self.__class__
        if hasattr(cls, 'identity'):
            empty[:, :] = torch.tensor(cls.identity)
        return torch.all(torch.isclose(self.tensor, empty), dim=-1)

    def __len__(self):
        return self.tensor.shape[0]

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to(self, device: str) -> 'AlignmentBase':
        obj = type(self)(self.tensor.to(device))
        return obj

    def clone(self) -> 'AlignmentBase':
        obj = type(self)(self.tensor.clone())
        return obj

    def __getitem__(self, idx) -> 'AlignmentBase':
        obj = type(self)(self.tensor[idx])
        return obj

    def __iter__(self):
        yield from self.tensor

    def __repr__(self):
        return '{}(tensor={})'.format(type(self).__name__, str(self.tensor))

    def split(self, sizes) -> List['AlignmentBase']:
        tensors = self.tensor.split(sizes)
        return [self.__class__(tensor=tensor) for tensor in tensors]

    @staticmethod
    def cat(mat_list) -> 'AlignmentBase':
        assert isinstance(mat_list, (list, tuple))
        assert len(mat_list) > 0
        dtype = type(mat_list[0])
        assert all(isinstance(mat, dtype) for mat in mat_list)

        cat_mats = dtype(tensor=cat([b.tensor for b in mat_list], dim=0))
        return cat_mats


class Scales(AlignmentBase):
    ndim = 3
    identity = [1, 1, 1]

    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor, Scales.ndim)


class Translations(AlignmentBase):
    ndim = 3
    identity = [0, 0, 0]

    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor, Translations.ndim)


class Rotations(AlignmentBase):
    ndim = 4
    identity = [1, 0, 0, 0]

    def __init__(self, tensor: torch.Tensor):
        # if not torch.allclose(tensor, torch.zeros_like(tensor)):
        super().__init__(tensor, Rotations.ndim)
        self.tensor = self.tensor / torch.norm(tensor, dim=1, keepdim=True)
        self.tensor = standardize_quaternion(self.tensor)

    @staticmethod
    def from_rotation_matrices(rotation_mats, use_pt3d=True) -> 'Rotations':
        if isinstance(rotation_mats, RotationMats):
            rotation_mats = rotation_mats.tensor.view(-1, 3, 3)
        else:
            assert isinstance(rotation_mats, torch.Tensor), \
                'rotation_mats must be a tensor or RotationsMats object'

        assert rotation_mats.ndim == 3 and rotation_mats.shape[-2:] == (3, 3)
        if use_pt3d:
            return Rotations(tensor=matrix_to_quaternion(rotation_mats))
        else:
            device = rotation_mats.device
            rotation_mats = rotation_mats.detach().cpu()
            quats = [
                quaternion.from_rotation_matrix(rm.numpy())
                for rm in rotation_mats.unbind(0)
            ]
            res = Rotations(tensor=torch.stack([
                torch.from_numpy(quaternion.as_float_array(q)) for q in quats
            ]))
            res = res.to(device)
            return res

    @torch.no_grad()
    def as_quaternions(self) -> List[np.quaternion]:
        tensor = self.tensor.cpu()
        quats = []
        for q in tensor.unbind(0):
            quats.append(np.quaternion(*q.tolist()))
        return quats

    def as_rotation_matrices(
        self,
        keep_device=True,
        use_pt3d=True
    ) -> 'RotationMats':
        if use_pt3d:
            mats = quaternion_to_matrix(self.tensor)
        else:
            quats = self.tensor.cpu()
            mats = []
            for q in quats.unbind(0):
                mats.append(torch.from_numpy(
                    quaternion.as_rotation_matrix(np.quaternion(*q.tolist()))
                ))
            mats = torch.stack(mats, dim=0).flatten(1)
            if keep_device:
                mats = mats.to(self.device)
        return RotationMats(tensor=mats)


class RotationMats(AlignmentBase):
    ndim = 9
    identity = [
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    ]

    def __init__(self, tensor: torch.Tensor):
        tensor = tensor.contiguous().flatten(1)
        super().__init__(tensor, RotationMats.ndim)

    @property
    def mats(self) -> torch.Tensor:
        return self.tensor.view(-1, 3, 3)
