from typing import List, Optional, Tuple, Type, Union

import torch

from roca.structures import AlignmentBase


def back_project(
    xy_grid: torch.Tensor,
    depth: torch.Tensor,
    intr: torch.Tensor,
    invert_intr: bool = True
) -> torch.Tensor:
    x = xy_grid[:, 0, :, :].flatten(1)
    y = xy_grid[:, 1, :, :].flatten(1)
    z = depth.flatten(1)
    if invert_intr:
        intr = intr.inverse()
    points = intr @ torch.stack([x * z, y * z, z], dim=1)
    return points.view(-1, 3, *depth.shape[-2:])


def depth_bbox(
    depth_points: torch.Tensor,
    depth: torch.Tensor,
    dummy: float = 1e3,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = depth <= eps
    flt = mask.flatten(1).all(-1, keepdim=True).logical_not_()
    max_coords = (depth_points - mask * dummy).flatten(2)\
        .max(-1)[0]
    min_coords = (depth_points + mask * dummy).flatten(2)\
        .min(-1)[0]
    return min_coords * flt, max_coords * flt


def depth_bbox_center(
    depth_points: torch.Tensor,
    depth: torch.Tensor,
    dummy: float = 1e6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_coords, max_coords = depth_bbox(depth_points, depth, dummy=dummy)
    return (max_coords + min_coords) / 2, min_coords, max_coords


def procrustes(
    nocs: torch.Tensor,
    depth_points: torch.Tensor,
    masks: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    trans: Optional[torch.Tensor] = None,
    mask_inputs: bool = True,
    ret_trans: bool = False,
    mask_probs: Optional[torch.Tensor] = None,
    zero_center: bool = False,
    prep_probs: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    # Pre-process inputs and convert them to batch of Dx3 matrices
    if trans is not None:
        depth_points = inverse_transform(depth_points, masks, trans=trans)
    elif mask_inputs:
        depth_points = depth_points * masks

    if scale is not None:
        nocs = transform(nocs, masks, scale=scale, mask_outputs=False)
    if mask_inputs:
        nocs = nocs * masks

    weighted = mask_probs is not None
    if weighted and prep_probs:
        mask_probs = mask_probs * masks
        mask_probs = mask_probs / mask_probs.sum((1, 2, 3), keepdim=True)

    if ret_trans:
        # Calculate the mean
        if weighted:
            noc_mean = (nocs * mask_probs).flatten(2).sum(-1)
            depth_mean = (depth_points * mask_probs).flatten(2).sum(-1)
        else:
            num_points = point_count(masks)
            noc_mean = point_mean(nocs, num_points)
            depth_mean = point_mean(depth_points, num_points)

        # Zero-center points
        if zero_center:
            nocs = (nocs - noc_mean.view(-1, 3, 1, 1)) * masks
            depth_points = (depth_points - depth_mean.view(-1, 3, 1, 1))\
                * masks

    depth_points = depth_points.flatten(2)
    nocs = nocs.flatten(2)

    # Get rotation
    if weighted:
        mask_probs = mask_probs.flatten(2)
        mat = depth_points @ (mask_probs * nocs).transpose(1, 2)
    else:
        mat = depth_points @ nocs.transpose(1, 2)
    r_pred = svd(mat)

    if ret_trans:
        rotated_noc_mean = r_pred.view(-1, 3, 3) @ noc_mean.view(-1, 3, 1)
        trs = depth_mean - rotated_noc_mean.view_as(depth_mean)
        return r_pred, trs

    return r_pred


def svd(mat: torch.Tensor) -> torch.Tensor:
    device = mat.device
    batch_size = mat.size(0)
    dtype = mat.dtype

    if mat.ndim == 2:
        mat = mat.reshape(batch_size, 3, 3)
    if dtype != torch.double: # important for numerical stability
        mat = mat.double()

    u, _, v = torch.svd(mat)

    # Handle reflection via determinant (1 or -1)
    d = torch.det(u @ v.transpose(1, 2))
    d = torch.cat([
        torch.ones(batch_size, 2, device=device),
        d.unsqueeze(-1)
    ], axis=-1)
    d = torch.eye(3, device=device).unsqueeze(0) * d.view(-1, 1, 3)

    # Predict and return a batch of 3x3 orthogonal matrices
    mat = u @ d @ v.transpose(1, 2)
    return mat.to(dtype)


def point_mean(pts: torch.Tensor, num_points: float) -> torch.Tensor:
    return pts.flatten(2).sum(-1) / num_points


def point_count(masks: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return masks.flatten(2).sum(-1).clamp(eps)


def residuals(
    s_nocs: torch.Tensor,
    it_depths: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    rot: Optional[torch.Tensor] = None,
    trans: Optional[torch.Tensor] = None,
    hard_weights: Optional[torch.Tensor] = None
    # hard_weights: torch.Tensor = None
) -> torch.Tensor:
    s_nocs = transform(s_nocs, masks, rot=rot)
    it_depths = inverse_transform(it_depths, masks, trans=trans)
    abs_diffs = (s_nocs - it_depths).norm(dim=1, keepdim=True)
    if hard_weights is not None:
        # FIXME: detach is a numerical stability hack!
        abs_diffs = abs_diffs * hard_weights.detach().sqrt()
    return abs_diffs


def irls(
    s_nocs: torch.Tensor,
    it_depths: torch.Tensor,
    masks: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    num_iter: int = 1,
    ret_history: bool = False
) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    assert num_iter >= 1

    hard_weighted = weights is not None
    hard_weights = weights if hard_weighted else None
    if not hard_weighted:
        weights = torch.ones_like(masks)

    error_history = []
    solution_history = []

    num_points = None

    for i in range(1, num_iter + 1):
        # Solve the least squares problem
        new_rot, new_trans = procrustes(
            s_nocs.double(),
            it_depths.double(),
            masks.double(),
            ret_trans=True,
            mask_probs=weights.double(),
            prep_probs=True
        )
        rot, trans = new_rot.float(), new_trans.float()

        # No weight update needed, last iteration!
        if i == num_iter and not ret_history:
            break

        # Compute error terms
        errors = residuals(s_nocs, it_depths, masks, rot, trans, hard_weights)

        # Record losses and solutions if demanded
        if ret_history:
            # TODO: add hard weights here
            if num_points is None:
                num_points = point_count(masks)
            error_history.append(point_mean(errors, num_points))
            solution_history.append((rot, trans))
            if i == num_iter:  # Last iteration, no update!
                break

        # Update weights
        errors = errors.clamp(1e-5)
        weights = (hard_weights if hard_weighted else masks) / errors

    # import pdb; pdb.set_trace()

    if ret_history:
        return solution_history, error_history
    else:
        return rot, trans


def transform(
    inputs: torch.Tensor,
    masks: torch.Tensor = None,
    scale: torch.Tensor = None,
    rot: torch.Tensor = None,
    trans: torch.Tensor = None,
    mask_outputs: bool = True
) -> torch.Tensor:
    outputs = inputs.flatten(2)
    if scale is not None:
        outputs = outputs * scale.unsqueeze(-1)  # (N, 3, D) * (N, 3, 1)
    if rot is not None:
        outputs = rot @ outputs  # (N, 3, 3) @ (N, 3, D)
    if trans is not None:
        outputs = outputs + trans.unsqueeze(-1)  # (N, 3, D) + (N, 3, 1)
    outputs = outputs.view_as(inputs)
    if mask_outputs and masks is not None:
        outputs = outputs * masks
    return outputs


def inverse_transform(
    inputs: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    rot: Optional[torch.Tensor] = None,
    trans: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Same dims as transform
    outputs = inputs.flatten(2)
    if trans is not None:
        outputs = outputs - trans.unsqueeze(-1)
    if rot is not None:
        outputs = rot.transpose(1, 2) @ outputs
    if scale is not None:
        outputs = outputs / scale.unsqueeze(-1)
    outputs = outputs.view_as(inputs)
    if masks is not None:
        outputs = outputs * masks
    return outputs


def reflect_x(
    points: torch.Tensor,
    x_reflection: torch.Tensor
) -> torch.Tensor:
    if not x_reflection.any():
        return points
    reflection = torch.ones(x_reflection.numel(), 3, device=points.device)
    reflection[x_reflection, 0] = -1
    return points * reflection.view(-1, 3, 1, 1)


def make_new(
    struct: Type[AlignmentBase],
    has_alignment: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if has_alignment.all():
        return tensor
    else:
        num_instances = has_alignment.numel()
        device = tensor.device
        result = struct.new_empty(num_instances, device=device).tensor
        result[has_alignment] = tensor
        return result
