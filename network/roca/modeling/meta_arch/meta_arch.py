from typing import Any, Dict, List, Optional, Union

import torch

from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

from roca.data.constants import VOXEL_RES
from roca.utils.misc import make_dense_volume


# Based on detectron2.GeneralizedRCNN
@META_ARCH_REGISTRY.register()
class ROCA(GeneralizedRCNN):
    def forward(self, batched_inputs: List[Any]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if 'instances' in batched_inputs[0]:
            gt_instances = [
                x['instances'].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        # Extract the image depth
        image_depths = []
        for input in batched_inputs:
            image_depths.append(input.pop('image_depth'))
        image_depths = torch.cat(image_depths, dim=0).to(self.device)

        # Run the network
        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert 'proposals' in batched_inputs[0]
            proposals = [
                x['proposals'].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances, image_depths
        )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Any],
        detected_instances: Optional[List[Instances]] =None,
        do_postprocess: bool = True
    ) -> Union[List[Instances], Dict[str, Any]]:
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert 'proposals' in batched_inputs[0]
                proposals = [
                    x['proposals'].to(self.device) for x in batched_inputs
                ]
            targets = [
                {'intrinsics': input['intrinsics'].to(self.device)}
                for input in batched_inputs
            ]
            scenes = [input['scene'] for input in batched_inputs]
            results, extra_outputs = self.roi_heads(
                images,
                features,
                proposals,
                targets=targets,
                scenes=scenes
            )
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            results = self.__class__._postprocess(
                results, batched_inputs, images.image_sizes
            )

        # Attach image depths
        if 'pred_image_depths' in extra_outputs:
            pred_image_depths = extra_outputs['pred_image_depths'].unbind(0)
            for depth, result in zip(pred_image_depths, results):
                result['pred_image_depth'] = depth
        
        # Attach CAD ids
        for cad_ids in ('cad_ids', 'wild_cad_ids'):
            if cad_ids in extra_outputs:
                # indices are global, so all instances should have all CAD ids
                for result in results:
                    result[cad_ids] = extra_outputs[cad_ids]

        return results

    @property
    def retrieval_head(self):
        return self.roi_heads.alignment_head.retrieval_head

    def set_train_cads(self, points, ids):
        retrieval_head = self.retrieval_head

        retrieval_head.wild_points_by_class = (
            {k: p.to(self.device) for k, p in points.items()}
            if retrieval_head.baseline
            else points
        )
        retrieval_head.wild_ids_by_class = ids

        self.train_cads_embedded = False

    def unset_train_cads(self):
        retrieval_head = self.retrieval_head
        retrieval_head.wild_points_by_class = None
        retrieval_head.wild_ids_by_class = None
        self.train_cads_embedded = False

    def embed_train_cads(self, batch_size: int = 16):
        return self._embed_cads(wild=True, batch_size=batch_size)

    def set_cad_models(self, points, ids, scene_data):
        self.retrieval_head.inject_cad_models(
            points=points,
            ids=ids,
            scene_data=scene_data,
            device=self.device if self.retrieval_head.baseline else 'cpu'
        )
        self.val_cads_embedded = False

    def unset_cad_models(self):
        self.retrieval_head.eject_cad_models()
        self.val_cads_embedded = False

    def embed_cad_models(self, batch_size: int = 16):
        return self._embed_cads(wild=False, batch_size=batch_size)

    @torch.no_grad()
    def _embed_cads(self, wild: bool = True, batch_size: int = 16):
        retrieval_head = self.retrieval_head
        if retrieval_head.baseline:
            return

        if wild:
            assert retrieval_head.has_wild_cads, \
                'Call `set_train_cads` before embedding cads'
            points_by_class = retrieval_head.wild_points_by_class
        else:
            assert retrieval_head.has_cads, \
                'Call `set_cad_models` before embedding cads'
            points_by_class = retrieval_head.points_by_class

        # Below makes this function callable twice!
        if wild and self.train_cads_embedded:
            return
        if not wild and self.val_cads_embedded:
            return

        is_voxel = self.retrieval_head.is_voxel
        for cat, points in points_by_class.items():
            embeds = []
            total_size = points.size(0) if not is_voxel else len(points)
            for i in range(0, total_size, batch_size):
                points_i = points[i:min(i + batch_size, total_size)]
                if is_voxel:
                    points_i = torch.stack([
                        make_dense_volume(p, VOXEL_RES) for p in points_i
                    ])
                embeds.append(
                    retrieval_head.embed_cads(points_i.to(self.device)).cpu()
                )

            points_by_class[cat] = torch.cat(embeds).to(self.device)
            del embeds

        if wild:
            self.train_cads_embedded = True
        else:
            self.val_cads_embedded = True

    def __getattr__(self, k):
        # Data dependency injections
        if 'inject' in k or 'eject' in k or k == 'set_verbose':
            return getattr(self.roi_heads, k)
        return super().__getattr__(k)
