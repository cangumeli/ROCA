from typing import Optional

import torch
import torch.nn.functional as F

from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.layers import nonzero_tuple
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
)


class WeightedFastRCNNOutputs(FastRCNNOutputs):
    def __init__(self, *args, **kwargs):
        try:
            self.class_weights = kwargs.pop('class_weights')
        except KeyError:
            self.class_weights = None
        super().__init__(*args, **kwargs)

    def softmax_cross_entropy_loss(self) -> torch.Tensor:
        if self.class_weights is None:
            return super().softmax_cross_entropy_loss()
        # import pdb; pdb.set_trace()
        weights = self.class_weights[self.gt_classes]
        # print('Passed...')
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            losses = F.cross_entropy(
                self.pred_class_logits, self.gt_classes, reduction='none'
            )
            # return losses
            return torch.sum(weights * losses) / torch.sum(weights)

    def box_reg_loss(self):
        if self.class_weights is None:
            return super().box_reg_loss()

        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
            cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
            device = self.pred_proposal_deltas.device

            bg_class_ind = self.pred_class_logits.shape[1] - 1

            # Box delta loss is only computed between the prediction for the gt class k
            # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
            # for non-gt classes and background.
            # Empty fg_inds produces a valid loss of zero as long as the size_average
            # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
            # and would produce a nan loss).
            fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
            if cls_agnostic_bbox_reg:
                # pred_proposal_deltas only corresponds to foreground class for agnostic
                gt_class_cols = torch.arange(box_dim, device=device)
            else:
                fg_gt_classes = self.gt_classes[fg_inds]
                # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
                # where b is the dimension of box representation (4 or 5)
                # Note that compared to Detectron1,
                # we do not perform bounding box regression for background classes.
                gt_class_cols = box_dim * fg_gt_classes[:, None]\
                    + torch.arange(box_dim, device=device)

            if self.box_reg_loss_type == "smooth_l1":
                gt_proposal_deltas = self.box2box_transform.get_deltas(
                    self.proposals.tensor, self.gt_boxes.tensor
                )
                loss_box_reg = smooth_l1_loss(
                    self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                    gt_proposal_deltas[fg_inds],
                    self.smooth_l1_beta,
                    reduction='none',
                )
                weights = self.class_weights[fg_gt_classes]\
                    .unsqueeze(1).expand_as(loss_box_reg)
            elif self.box_reg_loss_type == "giou":
                loss_box_reg = giou_loss(
                    self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                    self.gt_boxes.tensor[fg_inds],
                    reduction='none',
                )
                weights = self.class_weights[fg_gt_classes]
            else:
                raise ValueError(
                    f"Invalid bbox reg loss type '{self.box_reg_loss_type}'"
                )

            return torch.sum(loss_box_reg * weights) / torch.sum(weights)


class WeightedFastRCNNOutputLayers(FastRCNNOutputLayers):
    def set_class_weights(self, class_weights: Optional[torch.Tensor]):
        self.class_weights = class_weights

    def losses(self, predictions, proposals, class_weights=None):
        """See FastRCNNOutputLayers.losses"""
        class_weights = (
            self.class_weights if hasattr(self, 'class_weights') else None
        )
        if class_weights is None:
            return super().losses(predictions, proposals)

        scores, proposal_deltas = predictions
        losses = WeightedFastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            class_weights=class_weights
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
