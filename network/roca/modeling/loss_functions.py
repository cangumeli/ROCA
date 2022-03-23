from typing import Optional

import torch
import torch.nn.functional as F

import fvcore.nn as fvnn


def _apply_instance_average(losses, masks):
    losses = losses.flatten(1).sum(1) / masks.flatten(1).sum(1).clamp(1e-5)
    return losses.mean()


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mask_inputs: bool = True,
    weights: Optional[torch.Tensor] = None,
    instance_average: bool=False,
    log: bool = False
) -> torch.Tensor:

    if mask_inputs:
        pred = pred * mask
        target = target * mask

    losses = F.l1_loss(pred, target, reduction='none')
    if log:
        losses = torch.log(losses + 0.5) * mask

    if weights is not None:
        while weights.ndim < mask.ndim:
            weights = weights.unsqueeze(-1)
        mask = mask * weights
        losses = losses * weights

    if instance_average:
        return _apply_instance_average(losses, mask)
    else:
        return losses.sum() / mask.sum().clamp(1e-5)


# https://github.com/abduallahmohamed/reversehuberloss/blob/master/rhuloss.py
def inverse_huber_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mask_inputs: bool = True,
    weights: Optional[torch.Tensor] = None,
    instance_average: bool = False
) -> torch.Tensor:

    if mask_inputs:
        output = output * mask
        target = target * mask

    absdiff = torch.abs(output - target)
    C = 0.2 * torch.max(absdiff).item()
    losses = torch.where(
        absdiff < C,
        absdiff,
        (absdiff * absdiff + C * C) / (2 * C)
    )

    if weights is not None:
        while weights.ndim < mask.ndim:
            weights = weights.unsqueeze(-1)
        mask = mask * weights
        losses = losses * weights
    # import pdb; pdb.set_trace()
    if instance_average:
        return _apply_instance_average(losses, mask)
    else:
        return losses.sum() / mask.sum().clamp(1e-5)


def cosine_distance(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    dists = F.cosine_similarity(output, target, dim=1).view_as(mask)
    losses = (1 - dists) * mask
    if weights is not None:
        while weights.ndim < mask.ndim:
            weights = weights.unsqueeze(-1)
        mask = mask * weights
        losses = losses * weights
    return losses.sum() / mask.sum().clamp(1e-5)


def _apply_weights(losses, weights):
    if losses.ndim > weights.ndim:
        weights = weights.view(
            *weights.shape,
            *[1 for _ in range(losses.ndim - weights.ndim)]
        )
    if losses.numel() > weights.numel():
        weights = weights.expand_as(losses)
    return torch.sum(losses * weights) / torch.sum(weights)


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if weights is None:
        return F.l1_loss(pred, target, reduction='mean')
    else:
        losses = F.l1_loss(pred, target, reduction='none')
        return _apply_weights(losses, weights)


def l1_rel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    diffs = F.l1_loss(pred, target, reduction='none')
    rels = diffs / target
    if weights is None:
        return rels.mean()
    else:
        return _apply_weights(rels, weights)


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if weights is None:
        return F.mse_loss(pred, target, reduction='mean')
    else:
        losses = F.mse_loss(pred, target, reduction='none')
        return _apply_weights(losses, weights)


def l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    diffs = torch.norm(pred - target, dim=-1)
    if weights is None:
        return torch.mean(diffs)
    else:
        return _apply_weights(diffs, weights)


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    beta: float = 1.0
) -> torch.Tensor:
    diffs = fvnn.smooth_l1_loss(pred, target, reduction='none', beta=beta)
    if weights is None:
        return torch.mean(diffs)
    else:
        return _apply_weights(diffs, weights)


def binary_cross_entropy_with_logits(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if weights is None:
        return F.binary_cross_entropy_with_logits(
            pred, target, reduction='mean'
        )
    else:
        losses = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        return _apply_weights(losses, weights)


def mask_iou_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    gt, pred = gt.flatten(1), pred.flatten(1)
    gt_n0 = (gt.sum(-1) != 0).float()
    gt = gt.float()
    intr = pred * gt
    uni = pred + gt - intr
    iou = gt_n0 - intr.sum(-1) / uni.sum(-1).clamp(1e-5)
    if weights is None:
        return iou.mean()
    else:
        weights = weights.view_as(iou)
        return (weights * iou).sum() / weights.sum()


def angle_diff_loss(
    pred_mats: torch.Tensor,
    gt_mats: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    mats = pred_mats.transpose(1, 2) @ gt_mats
    trace = mats[:, 0, 0] + mats[:, 1, 1] + mats[:, 2, 2]
    acos = torch.acos(torch.clip(.5 * (trace - 1), min=-.99, max=.99))
    # if torch.isnan(acos).any():
    if weights is None:
        return acos.mean()
    else:
        return _apply_weights(acos, weights)
