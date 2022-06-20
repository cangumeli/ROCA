from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from pytorch3d.ops import add_pointclouds_to_volumes, knn_points, wmean
from pytorch3d.structures import Pointclouds, Volumes


def random_retrieval(cad_points: Tensor) -> Tuple[int, float]:
    index = torch.randint(
        low=0,
        high=cad_points.size(0),
        size=(1,)
    )[0]
    distance = Tensor(1.0, device=cad_points.device)
    return index, distance


def nearest_points_retrieval(
    points: Tensor,
    mask: Tensor,
    cad_points: Tensor,
    use_median: bool = False,
    mask_probs: Tensor = None,
    min_points: int = 20
) -> Tuple[int, float]:

    mask = mask > 0.5
    # Random sample if no points
    if mask.sum() <= min_points:
        return random_retrieval(cad_points)

    # Reshape to point clouds
    # 3, H, W -> H*W, 3
    points = points.flatten(1).t().contiguous()

    # 1, H, W -> H*W
    mask = mask.flatten()

    # Make points N, H*W
    points = points[mask]
    expand_size = (cad_points.size(0), *points.shape)
    points = points.unsqueeze(0).expand(*expand_size)
    assert points.ndim == cad_points.ndim
    assert points.size(0) == cad_points.size(0)

    # Get distance to nearest points and average
    distances, _, _ = knn_points(points, cad_points, K=1)  # N, D, 1
    distances = distances.squeeze(-1)
    if use_median:
        distances = distances.median(-1).values
    elif mask_probs is not None:
        mask_probs = mask_probs.flatten()[mask]
        distances = wmean(distances.t(), mask_probs, keepdim=False)
    else:
        distances = distances.mean(-1)

    # Return the closest CAD
    index = distances.argmin()
    distance = distances.min()
    return index, distance


def mask_point_features( x: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is not None and mask.numel() > 0:
        x = x * mask + x.min(-1, keepdim=True).values * mask.logical_not()
    return x


def pairwise_euclidian(x1: Tensor, x2: Tensor) -> Tensor:
    return x1.square().sum(1)[:, None] - 2 * x1.mm(x2.t()) + x2.square().sum(1)


def embedding_lookup(
    pred_classes: Tensor,
    noc_embeds: Tensor,
    cad_embeds_by_class: Dict[int, Tensor],
    cad_ids_by_class: Dict[int, Tuple[str, str]]
) -> List[Tuple[str, str]]:

    pred_classes = pred_classes.cpu()
    unique_classes = torch.unique(pred_classes).tolist()
    cad_ids = [None for _ in range(pred_classes.numel())]
    
    for c in unique_classes:
        if c not in cad_ids_by_class:
            continue

        class_mask = pred_classes == c
        noc_embeds_c = noc_embeds[class_mask]
        cad_embeds_c = cad_embeds_by_class[c]
        cad_ids_c = cad_ids_by_class[c]

        distances = pairwise_euclidian(noc_embeds_c, cad_embeds_c)
        indices = distances.argmin(-1).cpu().tolist()

        cad_stack = [cad_ids_c[i] for i in indices]
        cad_stack.reverse()
        for i, m in enumerate(class_mask.tolist()):
            if m:
                cad_ids[i] = cad_stack.pop()

    return cad_ids


def grid_to_point_list(
    nocs: Tensor,
    masks: Tensor
) -> List[Tensor]:
    result = []
    masks = masks.flatten(1).bool()
    nocs = nocs.flatten(2).transpose(1, 2).contiguous()
    for noc, mask in zip(nocs.unbind(0), masks.unbind(0)):
        result.append(noc[mask])
    return result


def pad_points(
    x: Union[Tensor, List[Tensor]]
) -> Tuple[Tensor, Optional[Tensor]]:
    if isinstance(x, list):
        max_points = max(xi.size(0) for xi in x)
        x_padded = []
        mask = []
        for xi in x:
            num_points = xi.size(0)
            if num_points == max_points:
                xi_padded = xi
                mask_i = torch.ones(xi.size(0), 1, device=xi.device)
            else:
                pad = torch.zeros(
                    max_points - num_points, 3, device=xi.device
                )
                xi_padded = torch.cat([xi, pad])
                mask_i = torch.cat([
                    torch.ones(num_points, 1, device=xi.device),
                    torch.zeros(pad.size(0), 1, device=xi.device),
                ])
            x_padded.append(xi_padded)
            mask.append(mask_i)
        x = torch.stack(x_padded).transpose(1, 2)
        mask = torch.stack(mask).transpose(1, 2)
    else:
        x = x.transpose(1, 2)
        mask = None
    return x, mask


def voxelize_nocs(points_list: List[Tensor], grid_size: int = 32) -> Tensor:
    points_list = [
        pts if pts.numel() > 0 else torch.zeros(1, 3, device=pts.device)
        for pts in points_list
    ]
    feats = [torch.ones(p.size(0), 1, device=p.device) for p in points_list]
    points = Pointclouds(points_list, features=feats)

    volume_size = (len(points_list), 1, grid_size, grid_size, grid_size)
    volumes = Volumes(
        densities=torch.zeros(volume_size, device=points.device),
        voxel_size=(1 / (grid_size - 1)),
        features=torch.zeros(volume_size, device=points.device)
    )
    volumes = add_pointclouds_to_volumes(points, volumes)
    # Normalize volumes
    # TODO: Try alternatives
    voxels = volumes.densities()
    voxels = voxels / voxels.sum((2, 3, 4), keepdim=True).clamp(1e-5)

    return voxels
