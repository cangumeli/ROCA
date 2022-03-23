from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import detectron2.layers as L
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import Boxes, ImageList, Instances

from roca.modeling.alignment_head import AlignmentHead
from roca.modeling.common_ops import create_xy_grids, select_classes
from roca.modeling.depth_head import DepthHead
from roca.modeling.logging_metrics import mask_metrics
from roca.modeling.loss_functions import (
    binary_cross_entropy_with_logits,
    mask_iou_loss,
)
from roca.modeling.roi_heads.fast_rcnn import WeightedFastRCNNOutputLayers
from roca.structures import Masks
from roca.utils.logging import LogWindow


@ROI_HEADS_REGISTRY.register()
class ROCAROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.log_window = LogWindow()
        self.set_verbose(False)

        self._init_class_weights(cfg)
        self._customize_box_head(cfg)
        self._init_depth_head(cfg)
        self._init_alignment_head(cfg)

        self.output_grid_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2

        self.test_min_score = cfg.MODEL.ROI_HEADS.CONFIDENCE_THRESH_TEST

    def _init_class_weights(self, cfg):
        class_weights = cfg.MODEL.CLASS_SCALES
        self.use_class_weights = bool(class_weights)
        if self.use_class_weights:
            class_weight_tensor = torch.zeros(1 + len(class_weights))
            for i, scale in class_weights:
                class_weight_tensor[i + 1] = scale
            class_weight_tensor[0] = torch.max(class_weight_tensor[1:])
            self.register_buffer('class_weights', class_weight_tensor)

    def _customize_box_head(self, cfg):
        if not self.use_class_weights:
            return
        del self.box_predictor
        self.box_predictor = WeightedFastRCNNOutputLayers(
            cfg, self.box_head.output_shape
        )

    def _init_depth_head(self, cfg):
        self.depth_head = DepthHead(cfg, self.in_features, self.log_window)

    def _init_alignment_head(self, cfg):
        self.alignment_head = AlignmentHead(
            cfg,
            self.num_classes,
            self.depth_head.out_channels,
            self.log_window
        )
        self.per_category_mask = cfg.MODEL.ROI_HEADS.PER_CATEGORY_MASK
        if self.per_category_mask:
            self.mask_head.predictor = nn.Conv2d(
                self.mask_head.deconv.out_channels,
                self.num_classes + 1,
                kernel_size=(1, 1)
            )

    @property
    def has_cads(self) -> bool:
        return self.alignment_head.has_cads

    def inject_log_window(self, window: Dict[str, List[float]]):
        self.log_window.inject_log_window(window)

    def eject_log_window(self):
        self.log_window.eject_log_window()

    def set_verbose(self, verbose: bool = True):
        self.verbose = verbose

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        gt_depths: Optional[torch.Tensor] = None,
        scenes: Optional[List[str]] = None
    ):
        image_size = images[0].shape[-2:]  # Assume single image size!
        del images
        if self.training:
            assert targets
            assert gt_depths is not None
            proposals = self.label_and_sample_proposals(proposals, targets)
        else:
            inference_args = targets  # Extra arguments for inference
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)

            depth_losses, depths, depth_features = self._forward_image_depth(
                features, gt_depths
            )
            losses.update(depth_losses)

            losses.update(self._forward_alignment(
                features,
                proposals,
                image_size,
                depths,
                depth_features,
                gt_depths=gt_depths
            ))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)

            pred_depths, depth_features = self._forward_image_depth(features)
            extra_outputs = {'pred_image_depths': pred_depths}

            pred_instances, alignment_outputs = self._forward_alignment(
                features,
                pred_instances,
                image_size,
                pred_depths,
                depth_features,
                inference_args=inference_args,
                scenes=scenes
            )
            extra_outputs.update(alignment_outputs)

            return pred_instances, extra_outputs

    def _forward_box(self, *args, **kwargs):
        if self.use_class_weights:
            self.box_predictor.set_class_weights(self.class_weights)
        return super()._forward_box(*args, **kwargs)

    def _forward_image_depth(
        self,
        features: Dict[str, torch.Tensor],
        depth_gt: Optional[torch.Tensor] = None
    ):
        return self.depth_head(features, depth_gt)

    def _forward_alignment(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
        image_size: Tuple[int, int],
        depths: torch.Tensor,
        depth_features: torch.Tensor,
        inference_args: Optional[Dict[str, Any]] = None,
        gt_depths: Optional[torch.Tensor] = None,
        scenes: Optional[List[str]] = None
    ):
        features = [features[f] for f in self.in_features]

        if self.training:
            return self._forward_alignment_train(
                features,
                instances,
                image_size,
                depths,
                depth_features,
                gt_depths
            )
        else:
            return self._forward_alignment_inference(
                features,
                instances,
                image_size,
                depths,
                depth_features,
                inference_args,
                scenes
            )

    def _forward_alignment_train(
        self,
        features: List[torch.Tensor],
        instances: List[Instances],
        image_size: Tuple[int, int],
        depths: torch.Tensor,
        depth_features: torch.Tensor,
        gt_depths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        losses = {}

        # Declare some useful variables
        instances, _ = select_foreground_proposals(
            instances, self.num_classes
        )
        proposal_boxes = [x.proposal_boxes for x in instances]
        if self.train_on_pred_boxes:
            for pb in proposal_boxes:
                pb.clip(image_size)
        features = self.mask_pooler(features, proposal_boxes)
        boxes = Boxes.cat(proposal_boxes)
        batch_size = features.size(0)
        gt_classes = L.cat([p.gt_classes for p in instances])

        # Get class weight for losses
        if self.use_class_weights:
            class_weights = self.class_weights[gt_classes + 1]
        else:
            class_weights = None

        # Create xy-grids for back-projection and cropping, respectively
        xy_grid, xy_grid_n = create_xy_grids(
            boxes,
            image_size,
            batch_size,
            self.output_grid_size
        )

        # Mask
        mask_losses, mask_probs, mask_pred, mask_gt = self._forward_mask(
            features,
            gt_classes,
            instances,
            xy_grid_n=xy_grid_n,
            class_weights=class_weights
        )
        losses.update(mask_losses)

        losses.update(self.alignment_head(
            instances=instances,
            depth_features=depth_features,
            depths=depths,
            gt_depths=gt_depths,
            image_size=image_size,
            gt_classes=gt_classes,
            class_weights=class_weights,
            xy_grid=xy_grid,
            xy_grid_n=xy_grid_n,
            mask_pred=mask_pred,
            mask_probs=mask_probs,
            mask_gt=mask_gt
        ))

        return losses

    def _forward_alignment_inference(
        self,
        features: List[torch.Tensor],
        instances: List[Instances],
        image_size: Tuple[int, int],
        depths: torch.Tensor,
        depth_features: torch.Tensor,
        inference_args: Dict[str, Any],
        scenes: Optional[List[str]] = None
    ) -> Tuple[List[Instances], Dict[str, Any]]:

        score_flt = [p.scores >= self.test_min_score for p in instances]
        instances = [p[flt] for p, flt in zip(instances, score_flt)]

        pred_classes = [x.pred_classes for x in instances]
        pred_boxes = [x.pred_boxes for x in instances]
        instance_sizes = [len(x) for x in instances]
        features = self.mask_pooler(features, pred_boxes)

        # Predict the mask
        pred_classes = L.cat(pred_classes)
        pred_mask_probs, pred_masks = self._forward_mask(
            features, pred_classes
        )

        # Predict alignments
        predictions, extra_outputs = self.alignment_head(
            instances=instances,
            depth_features=depth_features,
            depths=depths,
            image_size=image_size,
            mask_probs=pred_mask_probs,
            mask_pred=pred_masks,
            inference_args=inference_args,
            scenes=scenes
        )
        predictions['pred_masks'] = pred_mask_probs

        # Fill the instances
        for name, preds in predictions.items():
            for instance, pred in zip(instances, preds.split(instance_sizes)):
                setattr(instance, name, pred)
        return instances, extra_outputs

    def _forward_mask(
        self,
        features,
        classes,
        instances=None,
        xy_grid_n=None,
        class_weights=None
    ):
        mask_logits = self.mask_head.layers(features)
        if self.per_category_mask:
            mask_logits = select_classes(
                mask_logits,
                self.num_classes + 1,
                classes
            )

        if self.training:
            assert instances is not None
            assert xy_grid_n is not None

            losses = {}

            mask_probs = torch.sigmoid(mask_logits)
            mask_pred = mask_probs > 0.5

            mask_gt = Masks\
                .cat([p.gt_masks for p in instances])\
                .crop_and_resize_with_grid(xy_grid_n, self.output_grid_size)

            losses['loss_mask'] = binary_cross_entropy_with_logits(
                mask_logits, mask_gt, class_weights
            )
            losses['loss_mask_iou'] = mask_iou_loss(
                mask_probs, mask_gt, class_weights
            )

            # Log the mask performance and then convert mask_pred to float
            self.log_window.log_metrics(
                lambda: mask_metrics(mask_pred, mask_gt.bool())
            )
            mask_pred = mask_pred.to(mask_gt.dtype)

            return losses, mask_probs, mask_pred, mask_gt
        else:
            mask_probs = torch.sigmoid_(mask_logits)
            mask_pred = (mask_probs > 0.7).to(mask_probs.dtype)
            return mask_probs, mask_pred
