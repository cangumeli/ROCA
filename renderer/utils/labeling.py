import os
import numpy as np
import cv2 as cv
from pycocotools import mask as mask_util
from itertools import product
from collections import namedtuple


class LabelMerger:
    def __init__(
        self,
        scene_dir: str,
        image_name: str,
        taxonomy: list,
        label_ids: list,
        idx: np.ndarray,
        alignments: list,
        num_models: int,
        iou_thresh: float = 0.5,
        in_place: bool = True
    ):
        self.scene_dir = scene_dir
        self.image_name = image_name

        self.taxonomy = {t['shapenet']: t for t in taxonomy}
        self.label_map = {idx: name for idx, name in label_ids}
        self.alignments = {a['id']: a for a in alignments}
        self.num_models = num_models

        self.iou_thresh = iou_thresh
        self.idx = idx if in_place else idx.copy()

        self.bg_names = (
            'wall', 'floor', 'ceiling', 'otherstructure', 'otherprop'
        )
        self.bg_ids = [
            k for k, v in self.label_map.items()
            if v in self.bg_names
        ]

        self._load_instances()
        self._remove_invisible_cads()
        # self._match_instances()
        self._compute_results()

    def _load_instances(self):
        image_path = os.path.join(
            self.scene_dir, 'instance', self.image_name + '.png'
        )
        image = cv.imread(image_path, -1)
        # import pdb; pdb.set_trace()
        labels, instances = np.divmod(image, 1000)

        # Zero out the background labels
        for bg_id in self.bg_ids:
            bg_mask = labels != bg_id
            labels *= bg_mask
            instances *= bg_mask

        # Open this to use a reduced label set
        '''for label in np.unique(labels):
            if label != 0 and label.item() not in self.label_map:
                labels[labels == label] = 0
                instances[labels == label] = 0
        '''
        self.labels = labels
        self.label_names = set(
            self.label_map[label.item()]
            for label in np.unique(labels) if label != 0
        )
        '''if int(self.image_name) == 200:
            import pdb; pdb.set_trace()'''
        if not len(self.label_names):
            self.instances = []
            return

        self.instances = []
        instance_ids = np.unique(instances)
        instance_ids = instance_ids[np.nonzero(instance_ids)]

        # Filter out all instances with a scan2cad label
        label_names = set()
        for t in self.taxonomy.values():
            label_names.update(t['nyud'])

        for idx in instance_ids:
            mask = instances == idx
            instance_labels = labels * mask

            # Filter out with max frequency label
            freqs = [
                (label, np.sum(instance_labels == label))
                for label in np.unique(instance_labels) if label > 0
            ]
            # import pdb; pdb.set_trace()
            label = freqs[np.argmax([f[1] for f in freqs]).item()][0]
            mask *= (labels == label)

            # Filter out backgrounds and scan2cad labels
            label_name = self.label_map[label.item()]
            '''if int(self.image_name) == 200:
                import pdb; pdb.set_trace()'''
 
            if label_name in label_names:
                continue
            if label_name in self.bg_names:
                continue

            self.instances.append({
                'label': label,
                'mask': mask.astype(np.uint8),
                'label_name': label_name
            })

    def _check_match(self, mask, mask_label: int, cad_mask, cad_synset: str):
        # print(self.image_name, cad_synset, mask_label)

        # Check category matching
        '''cat_match = True
        try:
            tax = self.taxonomy[cad_synset]
        except KeyError:
            cat_match = False

        # import pdb; pdb.set_trace()
        if cat_match:
            label_name = self.label_map[mask_label]
            cat_match = label_name in tax['nyud']

        if not cat_match:
            return 0
        '''

        # Check bbox matching return the iou score
        mask_bbox = mask_util.toBbox(mask_util.encode(np.asfortranarray(mask)))
        cad_bbox = mask_util.toBbox(mask_util.encode(np.asfortranarray(
            cad_mask
        )))
        iou = mask_util.iou([mask_bbox], [cad_bbox], [0])
        # print(iou)
        return iou[0][0]

    def _match_instances(self):
        cad_instances = [
            ((self.idx == i).astype(np.uint8), self.alignments[i])
            for i in np.unique(self.idx) if i != 0
        ]

        scores = []
        for (i_cad, (cad_mask, cad_alignment)), (i, instance) in product(
               enumerate(cad_instances), enumerate(self.instances)
        ):
            score = self._check_match(
                instance['mask'], instance['label'],
                cad_mask, cad_alignment['catid_cad']
            )
            scores.append((i_cad, i, score))
        scores.sort(key=lambda x: x[-1], reverse=True)
        # import sys; sys.exit(0)

        to_remove = set()
        matched_cad = set()
        for i_cad, i, score in scores:
            if i not in to_remove and score > self.iou_thresh:
                to_remove.add(i)
                matched_cad.add(i_cad)

        # Remove instances with high overlap
        self.instances = [
            instance for i, instance in enumerate(self.instances)
            if i not in to_remove
        ]

        # Remove hidden cad models
        for i_cad, (_, alignment) in enumerate(cad_instances):
            if i_cad not in matched_cad:
                self.idx[alignment['id']] = 0

        # Push scannet instances to foreground
        '''for instance in self.instances:
            self.idx *= instance['mask']'''

        # Avoid overlap with cad labels
        '''idx_filter = (self.idx == 0)
        to_remove.clear()
        for i, instance in enumerate(self.instances):
            instance['mask'] *= idx_filter
            if not np.any(instance['mask'] != 0):
                to_remove.add(i)
        self.instances = [
            instance for i, instance in enumerate(self.instances) 
            if i not in to_remove
        ]'''

    def _remove_invisible_cads(self):
        # import pdb; pdb.set_trace()
        for cid in np.unique(self.idx):
            if cid == 0:
                continue
            synset = self.alignments[cid.item()]['catid_cad']
            names = self.taxonomy[synset]['nyud']
            # import pdb; pdb.set_trace()
            mask = self.idx == cid
            labels = self.labels * mask
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[np.nonzero(unique_labels)]
            label_freqs_id = {
                ul.item(): np.sum(labels == ul)
                for ul in unique_labels
            }
            label_freqs = {
                self.label_map[k]: v for k, v in label_freqs_id.items()
            }
            area = mask.sum()

            label_names = set(
                self.label_map[label.item()]
                for label in unique_labels if label != 0
            )
            # TODO: review the constant .25
            label_names = set(
                ln for ln in label_names if label_freqs[ln] > area * .25
            )
            valid_names = label_names.intersection(names)
            if not len(valid_names):
                self.idx[mask] = 0
            else:
                name = self.taxonomy[synset]['name']
                # TODO: make this configurable
                if name not in ('chair', 'cabinet', 'bin'):
                    continue
                # Filter out semantically incoherent pixels
                label_ids = [
                    k for k, v in self.label_map.items() if v in valid_names
                ]
                label_mask = np.zeros_like(mask)
                for lid in label_ids:
                    label_mask += labels == lid
                # import pdb; pdb.set_trace()
                self.idx *= np.logical_not(mask)
                self.idx += label_mask * cid

    def _compute_results(self):
        new_idx_start = self.num_models + 1
        idx = new_idx_start
        self.new_instances = []
        for instance in self.instances:
            self.new_instances.append({
                'id': idx,
                'label': instance['label'].item(),
                'label_name': instance['label_name']
            })
            self.idx[instance['mask'].astype(np.bool)] = idx
            idx += 1

    MergeResult = namedtuple(
        'MergeResult', field_names=['idx', 'new_instances']
    )

    @property
    def result(self) -> MergeResult:
        return LabelMerger.MergeResult(
            idx=self.idx, new_instances=self.new_instances
        )


class LabelFilter:
    def __init__(
        self,
        scene_dir: str,
        match_ids: list,
        num_models: int,
        taxonomy: list,
        label_ids: list,
        label_names: dict,
        in_place: bool = True,
        bg_threshold: float = .25
    ):
        self.scene_dir = scene_dir
        self.match_ids = set(match_ids)
        self.num_models = num_models
        self.in_place = in_place
        self.bg_threshold = bg_threshold

        self.taxonomy = {t['shapenet']: t for t in taxonomy}
        self.label_map = {idx: name for idx, name in label_ids}
        self.label_map_rev = {name: idx for idx, name in label_ids}
        self.label_names = label_names

    def _read_image(self, image_name: str):
        image_path = os.path.join(
            self.scene_dir, 'instance', image_name + '.png'
        )
        image = cv.imread(image_path, -1)
        assert image is not None, '{} not found!'.format(image_path)
        labels, instances = np.divmod(image, 1000)
        '''if image_name == '000600':
            import pdb; pdb.set_trace()'''
        instance_set = set(i for i in np.unique(instances) if i > 0)
        to_remove = self.match_ids.intersection(instance_set)
        for i in to_remove:
            instances[instances == i] = 0
        return labels, instances

    def _compute_idx(
            self,
            idx: np.ndarray,
            labels: np.ndarray,
            instances: np.ndarray,
            alignments
    ):
        if not self.in_place:
            idx = idx.copy()

        # Remove incoherent cad segments
        '''for cid in np.unique(idx):
            if cid == 0:
                continue
            synset = self.alignments[cid.item()]['catid_cad']
            names = set(self.taxonomy[synset]['nyud'])
            mask = idx == cid
            area = mask.sum()
            label = mask * labels
            matches = []
            for lid in np.unique(label):
                if lid == 0:
                    continue
                if self.label_map[lid] not in names:
                    continue
                label_mask = label == lid
                label_area = label_mask.sum()
                if (label_area / area) > self.bg_threshold:
                    matches.append((label_area, label_mask))

            if len(matches) == 0:
                idx[mask] = 0
            else:
                label_mask = matches.sort(key=lambda x: x[0], reverse=True)[0]
                idx = (1 - mask) * idx + label_mask * idx
        '''
        # idx += (idx == 0) * (instances + self.num_models)
        # Include additional labels
        new_instances = []
        idx_zero = idx == 0
        for object_id, label_name in self.label_names.items():
            object_id = int(object_id)
            if object_id not in self.match_ids:
                mask = np.logical_and(idx_zero, instances == object_id)
                # if object_id == 8 and self.image_name == '000600':
                #    import pdb; pdb.set_trace()
                if not np.any(mask):
                    continue
                new_instances.append({
                    'id': object_id + self.num_models,
                    'label_name': label_name,
                    'label': self.label_map_rev[label_name]
                })
                idx[mask] = new_instances[-1]['id']
        return idx, new_instances

    def __call__(self, idx: np.ndarray, image_name: str, alignments=None):
        labels, instances = self._read_image(image_name)
        self.image_name = image_name
        idx, new_instances = self._compute_idx(
            idx, labels, instances, alignments
        )
        return idx, new_instances


class AreaFilter:
    def __init__(self, ratio: float, in_place: bool = True):
        assert 0 <= ratio < 1
        self.area_ratio = ratio ** 2
        self.in_place = in_place

    def __call__(self, idx: np.ndarray):
        if self.area_ratio == 0:
            return idx

        if not self.in_place:
            idx = idx.copy()

        min_area = idx.size * self.area_ratio
        for i in np.unique(idx):
            if i != 0:
                mask = idx == i
                if mask.sum() < min_area:
                    idx[mask] = 0

        return idx
