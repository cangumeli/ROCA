from typing import Tuple, Union

import torch


def make_dense_volume(
    ind: torch.Tensor,
    voxel_res: Union[int, Tuple[int, int, int]]
) -> torch.Tensor:

    if isinstance(voxel_res, int):
        voxel_res = (voxel_res, voxel_res, voxel_res)

    grid = torch.zeros(voxel_res, dtype=torch.bool)
    grid[ind[:, 0], ind[:, 1], ind[:, 2]] = True
    return grid.unsqueeze(0)
