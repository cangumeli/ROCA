import copy
import os
import random

import cv2 as cv
import numpy as np
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper

from roca.data import CADCatalog
from roca.data.constants import VOXEL_RES
from roca.structures import (
    Depths,
    Intrinsics,
    Masks,
    Rotations,
    Scales,
    Translations,
)
from roca.utils.misc import make_dense_volume


class Mapper(DatasetMapper):
    def __init__(self, cfg, is_train: bool, dataset_name: str):
        super().__init__(cfg, is_train)
        self._cfg = cfg
        augment = cfg.INPUT.AUGMENT

        self.augmentations = T.AugmentationList([])
        if augment:
            self.augmentations = T.AugmentationList([
                T.RandomContrast(0.8, 1.25),
                T.RandomBrightness(0.8, 1.25),
                T.RandomSaturation(0.8, 1.25)
            ])

        self.dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)

        # TODO: Better way to check baseline mode!
        self.is_voxel = cfg.INPUT.CAD_TYPE == 'voxel'
        self.negative_sample = (
            cfg.MODEL.RETRIEVAL_ON
            and not cfg.MODEL.RETRIEVAL_BASELINE
            and self.is_train
        )
        if self.negative_sample:
            self._cad_points, self._cad_ids = self._points_and_ids(
                dataset_name, self.is_voxel
            )

        if not is_train:
            self._intrinsic_cache = {}

    @staticmethod
    def _points_and_ids(dataset_name: str, is_voxel=False):
        cad_manager = CADCatalog.get(dataset_name)
        cad_points, cad_ids = \
            cad_manager.batched_points_and_ids(volumes=is_voxel)

        cad_ids_out = {c: {} for c in cad_ids.keys()}
        for c, cad_ids_c in cad_ids.items():
            for i, cad_id in enumerate(cad_ids_c):
                cad_ids_out[c][cad_id] = i
        return cad_points, cad_ids_out

    def __call__(self, dataset_dict: dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # Load image labels
        splits = self._split_filename(dataset_dict)
        scene_name, image_name = self._get_names(splits)
        dataset_dict['scene'] = scene_name

        # Load image
        try:
            image = utils.read_image(
                dataset_dict['file_name'], format=self.image_format
            )
        except FileNotFoundError:  # 400k val
            dataset_dict['file_name'] = dataset_dict['file_name'].replace(
                image_name + '.jpg', str(int(image_name)) + '.jpg'
            )
            image = utils.read_image(
                dataset_dict['file_name'], format=self.image_format
            )
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            aug_input = T.AugInput(image[:, :, ::-1])
            transforms = self.augmentations(aug_input)
            image = aug_input.image[:, :, ::-1]
        else:
            transforms = T.TransformList([])

        dataset_dict['image'] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        image_shape = image.shape[:2]

        if not self.is_train:
            dataset_dict['intrinsics'] = self._load_intrinsics(splits)

        if not self.is_train:
            dataset_dict.pop('annotations', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            raw_annos = dataset_dict.get('annotations', [])
            masks, image_depth = self._load_renderings(
                scene_name, image_name, raw_annos
            )
            dataset_dict['image_depth'] = image_depth

            # detectron2 default mapper processing
            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop('annotations')
                if obj.get('iscrowd', 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos,
                image_shape,
                mask_format=self.instance_mask_format
            )

            # Custom logic
            instances.gt_masks = masks
            instances = self._set_alignments(instances, raw_annos)
            instances = self._set_intrinsics(instances, raw_annos)
            if self.negative_sample:
                instances = self._set_triplets(instances, raw_annos)
            # TODO: from detectron2, check this behavior
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict['instances'] = instances

        return dataset_dict

    @staticmethod
    def _split_filename(dataset_dict):
        file_name = dataset_dict['file_name']
        prefix = []
        if file_name.startswith(os.sep):
            prefix = [os.sep]
        return prefix + file_name.split(os.sep)

    @staticmethod
    def _get_names(splits):
        image_name = splits[-1].replace('.jpg', '')
        scene_name = splits[-3]
        return scene_name, image_name

    def _load_intrinsics(self, splits):
        assert not self.is_train
        path = os.path.join(*splits[:-2], 'intrinsics_color.txt')
        if path in self._intrinsic_cache:
            intrinsics = self._intrinsic_cache[path]
        else:
            with open(path) as f:
                intrinsics = torch.tensor([
                    [float(f) for f in line.strip().split()]
                    for line in f
                ])
            self._intrinsic_cache[path] = intrinsics
        return Intrinsics(tensor=intrinsics[:3, :3])

    def _load_renderings(self, scene_name, image_name, annos):
        cfg = self._cfg

        scene_dir = os.path.join(self._metadata.rendering_root, scene_name)
        image_name = image_name + '.png'

        instance_path = os.path.join(scene_dir, 'instance', image_name)
        depth_path = os.path.join(scene_dir, 'depth', image_name)

        depth = cv.imread(depth_path, -1)
        image_depth = Depths.decode(depth, cfg.INPUT.DEPTH_SCALE).tensor

        instance = cv.imread(instance_path, cv.IMREAD_GRAYSCALE)
        masks = []
        for ann in annos:
            mask = instance == ann['alignment_id']
            masks.append(Masks(mask[None, :, :]))
        else:  # No mask, fill with empty
            empty = torch.zeros(0, *instance.shape, dtype=torch.bool)
            masks.append(Masks(empty))

        return Masks.cat(masks), image_depth

    @staticmethod
    def _set_alignments(instances, annos):
        if len(annos) > 0:
            instances.gt_translations = Translations.cat([
                Translations(torch.tensor([a['t']])) for a in annos
            ])
            instances.gt_rotations = Rotations.cat([
                Rotations(torch.tensor([a['q']])) for a in annos
            ])
            instances.gt_scales = Scales.cat([
                Scales(torch.tensor([a['s']])) for a in annos
            ])
        return instances

    @staticmethod
    def _set_intrinsics(instances, annos):
        instances.gt_intrinsics = Intrinsics.cat([
            Intrinsics(torch.tensor(a['intrinsics'])[:3, :3])
            for a in annos
        ])
        return instances

    def _set_triplets(self, instances, annos):
        pos = []
        neg = []
        classes = instances.gt_classes.tolist()
        for ann, cat in zip(annos, classes):
            model_info = ann['model']
            cat_id = model_info['catid_cad']
            id = model_info['id_cad']

            cad_ids = self._cad_ids[cat]
            cad_points = self._cad_points[cat]

            pos_ind = cad_ids[cat_id, id]
            pos.append(cad_points[pos_ind])

            sample_inds = [i for i in range(len(cad_ids)) if i != pos_ind]
            neg_ind = random.choice(sample_inds)
            neg.append(cad_points[neg_ind])

        # Convert to dense if using Voxels
        if self.is_voxel:

            def to_dense(ind):
                return make_dense_volume(ind, VOXEL_RES)

            pos = list(map(to_dense, pos))
            neg = list(map(to_dense, neg))

        # Set GT CADs
        instances.gt_pos_cads = torch.stack(pos)
        instances.gt_neg_cads = torch.stack(neg)

        # Set indices for sampling
        instances.gt_annot_ids = torch.tensor(
            [ann['id'] for ann in annos], dtype=torch.long
        )

        return instances

    def _set_reg_points(self, instances, annos):
        points = []
        classes = instances.gt_classes.tolist()
        for ann, cat in zip(annos, classes):
            model_info = ann['model']
            cat_id = model_info['catid_cad']
            id = model_info['id_cad']

            cad_ids = self._reg_cad_ids[cat]
            cad_points = self._reg_cad_points[cat]

            pos_ind = cad_ids[cat_id, id]
            points.append(cad_points[pos_ind])

        instances.gt_reg_points = torch.stack(points)
        return instances
