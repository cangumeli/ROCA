from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from roca.modeling.depth_head.depth_modules import (
    DepthFeatures,
    DepthOutput,
    Sobel,
)
from roca.modeling.logging_metrics import depth_metrics
from roca.modeling.loss_functions import (
    cosine_distance,
    inverse_huber_loss,
    masked_l1_loss,
)


class DepthHead(nn.Module):
    def __init__(self, cfg, in_features: int, log_window):
        super().__init__()

        self.in_features = in_features
        self.log_window = log_window

        # FIXME: This calculation must change if resizing!
        # depth_width = cfg.INPUT.DEPTH_RES[-1]
        # up_ratio = depth_width / 160
        up_ratio = 4
        feat_size = tuple(d // up_ratio for d in cfg.INPUT.DEPTH_RES)

        self.fpn_depth_features = DepthFeatures(size=feat_size)
        self.fpn_depth_output = DepthOutput(
            self.fpn_depth_features.out_channels,
            up_ratio
        )
        if cfg.MODEL.DEPTH_GRAD_LOSSES:
            self.sobel = Sobel()

        # TODO: make this configurable
        self.use_rhu = True
        self.use_grad_losses = cfg.MODEL.DEPTH_GRAD_LOSSES
        self.use_batch_average = cfg.MODEL.DEPTH_BATCH_AVERAGE

    @property
    def out_channels(self) -> int:
        return self.fpn_depth_features.out_channels

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        depth_gt: Optional[torch.Tensor] = None
    ) -> Union[
            Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor]]:

        features = [features[f] for f in self.in_features]
        if self.training:
            losses = {}
            assert depth_gt is not None
            mask = depth_gt > 1e-5
            flt = mask.flatten(1).any(1)
            using_grad_losses = self.use_grad_losses

            if not flt.any():
                zero_loss = torch.tensor(0.0, device=mask.device)
                losses.update({
                    'loss_image_depth': zero_loss
                })
                if using_grad_losses:
                    losses.update({
                        'loss_grad_x': zero_loss.clone(),
                        'loss_grad_y': zero_loss.clone(),
                        'loss_normal': zero_loss.clone()
                    })
                depth_features = torch.zeros(
                    depth_gt.size(0),
                    self.fpn_depth_features.out_channels,
                    *self.fpn_depth_features.size,
                    device=depth_gt.device
                )
                depth_pred = torch.zeros_like(depth_gt)
                return losses, depth_pred, depth_features

            depth_features = self.fpn_depth_features(features)
            raw_depth_pred = self.fpn_depth_output(depth_features)

            mask = mask[flt]
            depth_pred = raw_depth_pred[flt] * mask
            depth_gt = depth_gt[flt] * mask

            loss_fn = inverse_huber_loss if self.use_rhu else masked_l1_loss
            batch_average = self.use_batch_average

            # Directly compare the depths
            losses['loss_image_depth'] = loss_fn(
                depth_pred,
                depth_gt,
                mask,
                mask_inputs=False,
                instance_average=batch_average
            )

            # Log depth metrics to tensorboard
            self.log_window.log_metrics(lambda: depth_metrics(
                depth_pred,
                depth_gt,
                mask,
                mask_inputs=False,
                pref='depth/image_'
            ))

            # Grad loss
            if using_grad_losses:
                gradx_pred, grady_pred = self.sobel(depth_pred).chunk(2, dim=1)
                gradx_gt, grady_gt = self.sobel(depth_gt).chunk(2, dim=1)
                losses['loss_grad_x'] = loss_fn(
                    gradx_pred,
                    gradx_gt,
                    mask,
                    mask_inputs=False
                )
                losses['loss_grad_y'] = loss_fn(
                    grady_pred,
                    grady_gt,
                    mask,
                    mask_inputs=False
                )

                # Normal consistency loss
                # https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/train.py
                ones = torch.ones_like(gradx_pred)
                normal_pred = torch.cat([-gradx_pred, -grady_pred, ones], 1)
                normal_gt = torch.cat([-gradx_gt, -grady_gt, ones], 1)
                losses['loss_normal'] = 5 * cosine_distance(
                    normal_pred, normal_gt, mask
                )

            return losses, raw_depth_pred, depth_features
        else:
            depth_features = self.fpn_depth_features(features)
            depth_pred = self.fpn_depth_output(depth_features)
            return depth_pred, depth_features
