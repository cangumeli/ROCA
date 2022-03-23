from typing import Tuple, Union

import torch

from detectron2.structures import Boxes

from roca.structures import MeshGrid


def create_xy_grids(
    boxes: Union[Boxes, torch.Tensor],
    image_size: Tuple[int, int],
    batch_size: int,
    output_grid_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    if not torch.is_tensor(boxes):
        # Assume detectron2.Boxes
        boxes = boxes.tensor
    device = boxes.device

    if boxes.numel() == 0:
        size = (0, 2, output_grid_size, output_grid_size)
        return tuple(torch.zeros(size, device=device) for _ in range(2))

    # Use cpu as interpolation is not parallel
    mesh_grid = MeshGrid(image_size[::-1], batch_size)
    xy_grid, xy_grid_n = mesh_grid.crop_and_resize_with_norm(
        boxes.cpu(), output_grid_size, use_interpolate=True
    )
    return xy_grid.to(device), xy_grid_n.to(device)


def select_classes(
    inputs: torch.Tensor,
    num_classes: int,
    classes: torch.Tensor
) -> torch.Tensor:
    num_instances = inputs.size(0)
    num_channels = inputs.size(1) // num_classes
    spatial_size = inputs.size()[2:]
    new_size = (num_instances, num_classes, num_channels, *spatial_size)
    return inputs.view(new_size)[torch.arange(num_instances), classes]
