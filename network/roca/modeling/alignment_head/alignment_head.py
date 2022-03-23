from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import detectron2.layers as L
from detectron2.structures import Boxes, Instances

from roca.modeling.alignment_head.alignment_modules import (
    Aggregator,
    MLP,
    SharedMLP,
)
from roca.modeling.alignment_head.alignment_ops import (
    back_project,
    depth_bbox,
    depth_bbox_center,
    inverse_transform,
    irls,
    make_new,
    point_count,
    point_mean,
    transform,
)
from roca.modeling.common_ops import create_xy_grids, select_classes
from roca.modeling.logging_metrics import depth_metrics
from roca.modeling.loss_functions import (
    l1_loss,
    l2_loss,
    masked_l1_loss,
    smooth_l1_loss,
)
from roca.modeling.retrieval_head import RetrievalHead
from roca.structures import (
    Depths,
    Intrinsics,
    Rotations,
    Scales,
    Translations,
)


class AlignmentHead(nn.Module):
    def __init__(self, cfg, num_classes: int, input_channels: int, log_window):
        super().__init__()

        self.num_classes = num_classes
        self.output_grid_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION * 2
        self.log_window = log_window

        # Init the shape feature aggregator
        input_size = input_channels
        shape_code_size = 512

        shared_net = SharedMLP(
            input_size,
            hidden_size=shape_code_size,
            num_hiddens=2
        )
        global_net = MLP(
            shared_net.out_channels,
            hidden_size=shape_code_size,
            num_hiddens=2
        )
        self.shape_encoder = Aggregator(shared_net, global_net)
        self.shape_code_drop = nn.Dropout(0.3)

        # Init the scale head
        self.scale_head = MLP(
            shape_code_size,
            3 * self.num_classes,
            num_hiddens=2
        )

        # Init translation head
        self.per_category_trans = cfg.MODEL.ROI_HEADS.PER_CATEGORY_TRANS

        self.trans_head = MLP(
            3 + shape_code_size,
            3 * (self.num_classes if self.per_category_trans else 1),
            num_hiddens=1
        )

        self.e2e = cfg.MODEL.ROI_HEADS.E2E
        self.use_noc_embedding = cfg.MODEL.ROI_HEADS.NOC_EMBED
        self.per_category_noc = cfg.MODEL.ROI_HEADS.PER_CATEGORY_NOC
        self.min_nocs = cfg.MODEL.ROI_HEADS.NOC_MIN
        self.use_noc_weights = cfg.MODEL.ROI_HEADS.NOC_WEIGHTS
        self.use_noc_weight_head = cfg.MODEL.ROI_HEADS.NOC_WEIGHT_HEAD and self.e2e
        self.use_noc_weight_skip = cfg.MODEL.ROI_HEADS.NOC_WEIGHT_SKIP and self.e2e
        self.irls_iters = cfg.MODEL.ROI_HEADS.IRLS_ITERS

        embed_size = 0
        if self.use_noc_embedding:
            self.noc_embed = nn.Embedding(self.num_classes, 32)
            embed_size = self.noc_embed.embedding_dim

        # code + depth_points + scale + embedding
        noc_code_size = shape_code_size + 3 + 3 + embed_size
        noc_output_size = (
            3 * self.num_classes
            if self.per_category_noc
            else 3
        )
        self.noc_head = SharedMLP(
            noc_code_size,
            num_hiddens=2,
            output_size=noc_output_size
        )

        if self.use_noc_weight_head:
            self.noc_weight_head = SharedMLP(
                # noc code + noc + mask prob
                noc_code_size + 3 + 1,
                num_hiddens=1,
                # hidden_size=512,
                output_size=self.num_classes,
                output_activation=nn.Sigmoid
            )

        # Initialize the retrieval head
        self.retrieval_head = RetrievalHead(cfg, shape_code_size)
        self.wild_retrieval = cfg.MODEL.WILD_RETRIEVAL_ON

    @property
    def has_cads(self) -> bool:
        return self.retrieval_head.has_cads

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_training(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_training(
        self,
        instances: List[Instances],
        depth_features: torch.Tensor,
        depths: torch.Tensor,
        gt_depths: torch.Tensor,
        image_size: Tuple[int, int],
        gt_classes: torch.Tensor,
        class_weights: Optional[torch.Tensor],
        xy_grid: torch.Tensor,
        xy_grid_n: torch.Tensor,
        mask_pred: torch.Tensor,
        mask_probs: torch.Tensor,
        mask_gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        losses = {}

        proposal_boxes = [x.proposal_boxes for x in instances]

        # Extract depth roi features
        shape_code = self._encode_shape(
            proposal_boxes,
            mask_pred,
            depth_features,
            image_size
        )

        # Extract depth roi regions
        instance_sizes = torch.tensor(
            [len(p) for p in instances], device=self.device
        )
        intrinsics = Intrinsics.cat(
            [p.gt_intrinsics for p in instances]
        ).tensor.inverse()

        depth_losses, depth_pred, depth_gt = self._forward_roi_depth(
            xy_grid,
            depths,
            mask_pred,
            intrinsics,
            xy_grid_n,
            instance_sizes,
            gt_masks=mask_gt,
            gt_depths=gt_depths,
            class_weights=class_weights
        )
        depth_pred, depth_points_pred = depth_pred
        depth_gt, depth_points_gt = depth_gt
        losses.update(depth_losses)

        # Scale
        scale_losses, scale_pred, scale_gt = self._forward_scale(
            shape_code,
            gt_classes,
            instances=instances,
            class_weights=class_weights
        )
        losses.update(scale_losses)

        # Translation
        trans_losses, trans_pred, trans_gt = self._forward_trans(
            shape_code,
            depth_pred,
            depth_points_pred,
            gt_classes,
            instances=instances,
            gt_depth_points=depth_points_gt,
            gt_depths=depth_gt,
            class_weights=class_weights
        )
        losses.update(trans_losses)

        # Rotation
        rot_gt = Rotations.cat([p.gt_rotations for p in instances])
        proc_losses, _, nocs = self._forward_proc(
            shape_code,
            depth_points_pred,
            mask_pred,
            trans_pred,
            scale_pred,
            gt_classes,
            mask_probs=mask_probs,
            gt_depth_points=depth_points_gt,
            gt_masks=mask_gt,
            gt_trans=trans_gt,
            gt_rot=rot_gt.tensor,
            gt_scale=scale_gt,
            class_weights=class_weights
        )
        losses.update(proc_losses)

        # Retrieval
        losses.update(self._forward_retrieval_train(
            instances,
            mask_pred,
            nocs,
            shape_code
        ))

        return losses

    def forward_inference(
        self,
        instances: List[Instances],
        depth_features: torch.Tensor,
        depths: torch.Tensor,
        image_size: torch.Tensor,
        mask_probs: torch.Tensor,
        mask_pred: torch.Tensor,
        inference_args: List[Dict[str, Any]],
        scenes: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:

        instance_sizes = [len(x) for x in instances]
        num_instances = sum(instance_sizes)
        if num_instances == 0:
            return self.identity()

        predictions = {}

        pred_classes = [x.pred_classes for x in instances]
        pred_boxes = [x.pred_boxes for x in instances]

        # Filter instances based on has_alignment
        pred_masks = mask_pred
        pred_mask_probs = mask_probs
        pred_classes = L.cat(pred_classes)

        # Pool the shape features
        shape_code = self._encode_shape(
            pred_boxes,
            pred_masks,
            depth_features,
            image_size
        )

        pred_scales = self._forward_scale(shape_code, pred_classes)
        predictions['pred_scales'] = pred_scales

        # Predict translation and perform procrustes
        # if has_alignment.any():
        alignment_sizes = torch.tensor(instance_sizes, device=self.device)

        # Get the intrinsics
        # NOTE: Broadcast should work if batch size = 1
        intrinsics = Intrinsics.cat(
            [arg['intrinsics'] for arg in inference_args]
        ).tensor.inverse()
        if intrinsics.size(0) > 1:
            intrinsics = intrinsics.repeat_interleave(
                alignment_sizes, dim=0
            )

        xy_grid, xy_grid_n = create_xy_grids(
            Boxes.cat(pred_boxes),
            image_size,
            num_instances,
            self.output_grid_size
        )
        depths, depth_points, raw_depth_points = self._forward_roi_depth(
            xy_grid,
            depths,
            pred_masks,
            intrinsics,
            xy_grid_n,
            alignment_sizes
        )
        pred_transes = self._forward_trans(
            shape_code,
            depths,
            depth_points,
            pred_classes
        )
        predictions['pred_translations'] = pred_transes

        pred_rots, _, pred_nocs = self._forward_proc(
            shape_code,
            raw_depth_points,
            pred_masks,
            pred_transes,
            pred_scales,
            pred_classes,
            mask_probs=pred_mask_probs
        )
        predictions['pred_rotations'] = pred_rots

        if pred_nocs is not None:
            pred_nocs *= (pred_mask_probs > 0.5)  # Keep all foreground NOCs!

        # Do the retrieval
        has_alignment = torch.ones(sum(instance_sizes), dtype=torch.bool)
        predictions['has_alignment'] = has_alignment

        predictions, extra_outputs = self._forward_retrieval_inference(
            predictions=predictions,
            extra_outputs={},
            scenes=scenes,
            has_alignment=has_alignment,
            instance_sizes=instance_sizes,
            pred_scales=pred_scales,
            pred_transes=pred_transes,
            pred_rots=pred_rots,
            depth_points=depth_points,
            pred_nocs=pred_nocs,
            pred_masks=pred_masks,
            pred_classes=pred_classes,
            shape_code=shape_code
        )

        return predictions, extra_outputs

    def identity(self):
        if self.training:
            return {}
        else:
            device = next(self.parameters()).device
            predictions = {
                'pred_scales': Scales.new_empty(0, device).tensor,
                'pred_translations': Translations.new_empty(0, device).tensor,
                'pred_rotations': Rotations.new_empty(0, device).tensor,
                'has_alignment': torch.zeros(0, device=device).bool(),
                'pred_indices': torch.zeros(0, device=device).long()
            }
            extra_outputs = {'cad_ids': []}
            return predictions, extra_outputs

    def _encode_shape(
        self,
        pred_boxes,
        pred_masks,
        depth_features,
        image_size
    ):
        scaled_boxes = []
        for b in pred_boxes:
            b = Boxes(b.tensor.detach().clone())
            b.scale(
                depth_features.shape[-1] / image_size[-1],
                depth_features.shape[-2] / image_size[-2]
            )
            scaled_boxes.append(b.tensor)

        shape_features = L.roi_align(
            depth_features,
            scaled_boxes,
            self.output_grid_size
        )
        shape_code = self.shape_encoder(shape_features, pred_masks)
        shape_code = self.shape_code_drop(shape_code)
        return shape_code

    def _forward_scale(
        self,
        shape_code,
        alignment_classes,
        instances=None,
        class_weights=None
    ):
        scales = select_classes(
            self.scale_head(shape_code),
            self.num_classes,
            alignment_classes
        )

        if self.training:
            assert instances is not None
            losses = {}
            gt_scales = Scales.cat([p.gt_scales for p in instances]).tensor
            losses['loss_scale'] = l1_loss(
                scales,
                gt_scales,
                weights=class_weights
            )
            return losses, scales, gt_scales
        else:
            return scales

    def _forward_roi_depth(
        self,
        xy_grid,
        depths,
        pred_masks,
        intrinsics_inv,
        xy_grid_n,
        alignment_sizes,
        gt_masks=None,
        gt_depths=None,
        class_weights=None
    ):
        depths, depth_points = self._crop_and_project_depth(
            xy_grid,
            depths,
            intrinsics_inv,
            xy_grid_n,
            alignment_sizes
        )
        if self.training:
            assert gt_masks is not None
            assert gt_depths is not None

            losses = {}

            gt_depths, gt_depth_points = self._crop_and_project_depth(
                xy_grid,
                gt_depths,
                intrinsics_inv,
                xy_grid_n,
                alignment_sizes
            )

            losses['loss_roi_depth'] = masked_l1_loss(
                depths,
                gt_depths,
                gt_masks,
                weights=class_weights
            )

            # Log metrics to tensorboard
            self.log_window.log_metrics(lambda: depth_metrics(
                depths,
                gt_depths,
                gt_masks,
                pref='depth/roi_'
            ))

            # Mask the output depths for further use
            depths = depths * pred_masks
            depth_points = depth_points * pred_masks
            gt_depths = gt_depths * gt_masks
            gt_depth_points = gt_depth_points * gt_masks

            # Penalize depth means
            # TODO: make this loss optional
            depth_mean_pred = point_mean(depths, point_count(pred_masks))
            depth_mean_gt = point_mean(gt_depths, point_count(gt_masks))
            losses['loss_mean_depth'] = l1_loss(
                depth_mean_pred,
                depth_mean_gt,
                weights=class_weights
            )

            return losses, (depths, depth_points), (gt_depths, gt_depth_points)
        else:
            raw_depth_points = depth_points.clone()
            depths *= pred_masks
            depth_points *= pred_masks
            return depths, depth_points, raw_depth_points

    def _crop_and_project_depth(
        self,
        xy_grid,
        depths,
        intrinsics_inv,
        xy_grid_n,
        alignment_sizes
    ):
        depths = Depths(depths, alignment_sizes)\
            .crop_and_resize(
                xy_grid_n,
                crop_size=self.output_grid_size,
                use_grid=True
            )
        depth_points = back_project(
            xy_grid,
            depths,
            intrinsics_inv,
            invert_intr=False
        )
        return depths, depth_points

    def _forward_trans(
        self,
        shape_code,
        depths,
        depth_points,
        alignment_classes,
        instances=None,
        gt_depth_points=None,
        gt_depths=None,
        class_weights=None
    ):
        depth_center, depth_min, depth_max = depth_bbox_center(
            depth_points, depths
        )
        trans_code = L.cat(
            [(depth_max - depth_min).detach(), shape_code],
            dim=-1
        )
        trans_offset = self.trans_head(trans_code)
        if self.per_category_trans:
            trans_offset = select_classes(
                trans_offset,
                self.num_classes,
                alignment_classes
            )
        trans = depth_center + trans_offset

        if self.training:
            assert instances is not None
            assert gt_depth_points is not None
            assert gt_depths is not None

            losses = {}

            depth_min_gt, depth_max_gt = depth_bbox(gt_depth_points, gt_depths)
            losses['loss_depth_min'] = smooth_l1_loss(
                depth_min,
                depth_min_gt,
                weights=class_weights
            )
            losses['loss_depth_max'] = smooth_l1_loss(
                depth_max,
                depth_max_gt,
                weights=class_weights
            )

            trans_gt = Translations.cat([p.gt_translations for p in instances])
            trans_gt = trans_gt.tensor
            losses['loss_trans'] = l2_loss(
                trans,
                trans_gt,
                weights=class_weights
            )

            return losses, trans, trans_gt
        else:  # inference
            return trans

    def _forward_proc(
        self,
        shape_code,
        depth_points,
        masks,
        trans,
        scale,
        alignment_classes,
        mask_probs=None,
        gt_depth_points=None,
        gt_masks=None,
        gt_trans=None,
        gt_rot=None,
        gt_scale=None,
        class_weights=None
    ):
        # Untranslate depth using trans
        # depth_points = inverse_transform(depth_points, masks, trans=trans)
        depth_points = inverse_transform(depth_points, trans=trans)

        # Compute the nocs
        noc_codes = self._encode_shape_grid(
            shape_code,
            depth_points,
            scale,
            alignment_classes
        )
        nocs = self.noc_head(noc_codes)
        if self.per_category_noc:
            nocs = select_classes(nocs, self.num_classes, alignment_classes)
        raw_nocs = nocs
        nocs = masks * nocs

        # Perform procrustes steps to sufficiently large regions
        has_enough = masks.flatten(1).sum(-1) >= self.min_nocs
        do_proc = has_enough.any()
        if do_proc:
            rot, trs = self._solve_proc(
                nocs,
                depth_points,
                noc_codes,
                alignment_classes,
                has_enough,
                scale,
                masks,
                mask_probs
            )
        else:  # not do_proc
            rot, trs = None, None

        if self.training:
            assert gt_depth_points is not None
            assert gt_masks is not None
            assert gt_trans is not None
            assert gt_rot is not None
            assert gt_scale is not None

            losses = {}

            gt_rot = Rotations(gt_rot).as_rotation_matrices()
            gt_nocs = inverse_transform(
                gt_depth_points,
                gt_masks,
                gt_scale,
                gt_rot.mats,
                gt_trans
            )

            losses['loss_noc'] = 3 * masked_l1_loss(
                nocs,
                gt_nocs,
                torch.logical_and(masks, gt_masks),
                weights=class_weights
            )

            if self.e2e and do_proc:
                if class_weights is not None:
                    class_weights = class_weights[has_enough]

                losses['loss_proc'] = 2 * l1_loss(
                    rot.flatten(1),
                    gt_rot.tensor[has_enough],
                    weights=class_weights
                )
                losses['loss_trans_proc'] = l2_loss(
                    trs + trans.detach()[has_enough],
                    gt_trans[has_enough],
                    weights=class_weights
                )

            return losses, rot, nocs
        else:
            # Make the standard Procrustes inference
            if do_proc:
                trans[has_enough] += trs
                rot = Rotations.from_rotation_matrices(rot).tensor
                rot = make_new(Rotations, has_enough, rot)
            else:
                device = nocs.device
                batch_size = has_enough.numel()
                rot = Rotations.new_empty(batch_size, device=device).tensor

            return rot, nocs, raw_nocs

    def _encode_shape_grid(self, shape_code, depth_points, scale, classes):
        if self.use_noc_embedding:
            shape_code = L.cat([shape_code, self.noc_embed(classes)], dim=-1)

        shape_code_grid = shape_code\
            .view(*shape_code.size(), 1, 1)\
            .expand(*shape_code.size(), *depth_points.size()[-2:])
        scale_grid = scale.view(-1, 3, 1, 1).expand_as(depth_points)

        return L.cat([
            shape_code_grid,
            scale_grid.detach(),
            depth_points.detach()
        ], dim=1)

    def _solve_proc(
        self,
        nocs,
        depth_points,
        noc_codes,
        alignment_classes,
        has_enough,
        scale,
        masks,
        mask_probs
    ):
        proc_masks = masks[has_enough]
        s_nocs = transform(nocs[has_enough], scale=scale[has_enough])
        mask_probs = self._prep_mask_probs(
            mask_probs,
            noc_codes,
            nocs,
            alignment_classes,
            has_enough
        )
        return irls(
            s_nocs,
            depth_points[has_enough],
            proc_masks,
            weights=mask_probs,
            num_iter=self.irls_iters
        )

    def _prep_mask_probs(
        self,
        mask_probs,
        noc_codes,
        nocs,
        alignment_classes,
        has_enough=None
    ):
        if self.use_noc_weights:
            assert mask_probs is not None
            if has_enough is not None:
                mask_probs = mask_probs[has_enough]
                noc_codes = noc_codes[has_enough]
                nocs = nocs[has_enough]
                alignment_classes = alignment_classes[has_enough]

            if self.use_noc_weight_head:
                weight_inputs = [noc_codes, nocs.detach(), mask_probs.detach()]

                new_probs = select_classes(
                    self.noc_weight_head(L.cat(weight_inputs, dim=1)),
                    self.num_classes,
                    alignment_classes
                )

                # mask_probs = (1 - interp) * mask_probs + interp * new_probs
                if self.use_noc_weight_skip:
                    mask_probs = (mask_probs + new_probs) / 2
                else:
                    mask_probs = new_probs
        else:  # not self.use_noc_weights
            mask_probs = None

        return mask_probs

    def _forward_retrieval_train(self, instances, mask, nocs, shape_code):
        losses = {}

        if not self.retrieval_head.baseline:
            pos_cads = L.cat([p.gt_pos_cads for p in instances])
            neg_cads = L.cat([p.gt_neg_cads for p in instances])

            # TODO: make this configurable
            sample = torch.randperm(pos_cads.size(0))[:32]

            losses.update(self.retrieval_head(
                masks=mask[sample],
                noc_points=nocs[sample],
                shape_code=shape_code[sample],
                pos_cads=pos_cads[sample],
                neg_cads=neg_cads[sample]
            ))

        return losses

    def _forward_retrieval_inference(
        self,
        predictions,
        extra_outputs,
        scenes,
        has_alignment,
        instance_sizes,
        pred_scales,
        pred_transes,
        pred_rots,
        depth_points,
        pred_nocs,
        pred_masks,
        pred_classes,
        shape_code
    ):
        if self.has_cads:
            assert scenes is not None

            if pred_nocs is not None:
                noc_points = pred_nocs
            else:
                rotation_mats = Rotations(pred_rots)\
                    .as_rotation_matrices()\
                    .mats
                noc_points = inverse_transform(
                    depth_points,
                    pred_masks,
                    pred_scales,
                    rotation_mats,
                    pred_transes
                )

            cad_ids, pred_indices = self.retrieval_head(
                scenes=scenes,
                instance_sizes=instance_sizes,
                has_alignment=has_alignment,
                classes=pred_classes,
                masks=pred_masks,
                noc_points=noc_points,
                shape_code=shape_code
            )
            extra_outputs['cad_ids'] = cad_ids
            predictions['pred_indices'] = pred_indices

        if self.wild_retrieval:
            wild_cad_ids, wild_pred_indices = self.retrieval_head(
                scenes=scenes,
                instance_sizes=instance_sizes,
                classes=pred_classes,
                masks=pred_masks,
                noc_points=noc_points,
                wild_retrieval=self.wild_retrieval,
                shape_code=shape_code
            )
            extra_outputs['wild_cad_ids'] = wild_cad_ids
            predictions['pred_wild_indices'] = wild_pred_indices

        return predictions, extra_outputs
