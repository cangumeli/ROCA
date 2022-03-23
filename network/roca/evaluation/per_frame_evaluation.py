import json
import os
from collections import defaultdict, OrderedDict
from itertools import chain
from typing import Any, Dict, List, OrderedDict as OrderedDictType

import cv2 as cv
import numpy as np
import pycocotools.mask as mask_util
import torch
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou

from roca.data import CADCatalog, CategoryCatalog
from roca.modeling.loss_functions import masked_l1_loss
from roca.structures import Depths
from roca.utils.ap import compute_ap


class InstanceEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name: str, cfg, thresh=0.5):
        super().__init__()
        self.ap_fields = ('box', 'mask')
        self.output_dir = cfg.OUTPUT_DIR
        self._metadata = MetadataCatalog.get(dataset_name)
        self._category_manager = CategoryCatalog.get(dataset_name)
        self._val_cad_manager = CADCatalog.get(dataset_name)
        self._train_cad_manager = CADCatalog.get(cfg.DATASETS.TRAIN[0])
        self.pred_file = os.path.join(self.output_dir, 'per_frame_preds.json')
        self.thresh = thresh

        if cfg.MODEL.WILD_RETRIEVAL_ON:
            self.ap_fields = (*self.ap_fields, 'mesh')

    def reset(self):
        self.preds = {}
        self.step = 0

    def process(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]]
    ):
        self.step += 1
        for input, output in zip(inputs, outputs):
            self._add_preds(input, output)

        '''if self.step > 100:
            self.evaluate()
            exit(0)'''

    def evaluate(
        self,
        from_file: str = '',
        print_every: int = 500
    ) -> OrderedDictType[str, Dict[str, float]]:

        # Load predictions
        if from_file != '':
            self.pred_file = from_file
            all_preds = self._load_preds()
        else:
            self._save_preds()
            all_preds = self.preds

        # Collect results
        per_class_ap_data = self._eval_loop(all_preds, print_every)

        # Compute the global (instance) APs
        gAPs = {}
        for f, ap_data in per_class_ap_data.items():
            ap_values = ap_data.values()
            scores = chain(*(v['scores'] for v in ap_values))
            labels = chain(*(v['labels'] for v in ap_values))
            scores, labels = map(torch.as_tensor, map(list, (scores, labels)))
            npos = sum(v['npos'] for v in ap_values)
            gAPs[f] = np.round(
                compute_ap(scores, labels, npos).item() * 100,
                decimals=2
            ).item()

        # Compute per-category APs
        per_class_aps = {f: {} for f in per_class_ap_data.keys()}
        for f, ap_dict in per_class_aps.items():
            for cat, ap_data in per_class_ap_data[f].items():
                ap_dict[cat] = compute_ap(
                    torch.as_tensor(ap_data['scores']),
                    torch.as_tensor(ap_data['labels']),
                    ap_data['npos']
                ).item()

        # Average and report category APs
        mAPs = {}
        for f, v in per_class_aps.items():
            self.print('\nPer-Category Results for "{}"\n'.format(f.capitalize()))
            tab_data = list(v.items())
            tab_data = [
                (k, np.round(ap * 100, decimals=2)) for k, ap in tab_data
            ]
            tab_data.sort(key=lambda x: x[0])
            self.print(tabulate(
                tab_data,
                tablefmt='github',
                headers=['Category', 'AP']
            ))
            mAPs[f] = np.round(
                np.mean([ap for _, ap in tab_data]),
                decimals=2
            ).item()
            self.print()

        # Report and return mAPs
        for name, result in zip(['Mean', 'Instance'], [mAPs, gAPs]):
            self.print('\n{} APs Per Task\n'.format(name))
            self.print(tabulate(
                [(k.capitalize(), v) for k, v in result.items()],
                tablefmt='github',
                headers=['Task', 'mAP']
            ))
            self.print()

        return OrderedDict({
            f: {'mAP': mAPs[f], 'gAP': gAPs[f]}
            for f in per_class_aps.keys()
        })

    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs, flush=True)

    def _add_preds(self, input, output):
        instances = output['instances'].to('cpu')
        objects = []
        for i in range(len(instances)):
            # Register trivial predictions
            datum = {
                'score': instances.scores[i].item(),
                'bbox': instances.pred_boxes.tensor[i].tolist(),
                't': instances.pred_translations[i].tolist(),
                'q': instances.pred_rotations[i].tolist(),
                's': instances.pred_scales[i].tolist()
            }

            # Register classes
            class_name = self._category_manager.get_name(
                instances.pred_classes[i].item()
            )
            datum['category'] = class_name

            # Register rle mask
            mask = instances.pred_masks[i].numpy().squeeze()
            rle = mask_util.encode(np.array(
                mask[:, :, None], order='F', dtype='uint8'
            ))[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            datum['segmentation'] = rle

            # Register CAD retrievals
            if 'wild_cad_ids' in output:
                index = instances.pred_wild_indices[i].item()
                cad_id = output['wild_cad_ids'][index]
                datum['wild_cad_id'] = cad_id

            if 'cad_ids' in output:
                index = instances.pred_indices[i].item()
                cad_id = output['cad_ids'][index]
                datum['scene_cad_id'] = cad_id

            # Add predictions
            objects.append(datum)

        parts = '/'.join(input['file_name'].split('/')[-3:])
        self.preds[parts] = objects

    def _save_preds(self):
        with open(self.pred_file, 'w') as f:
            json.dump(self.preds, f)
    
    def _load_preds(self) -> dict:
        with open(self.pred_file) as f:
            preds = json.load(f)
        return preds

    def _parse_data_json(self) -> tuple:
        json_file = self._metadata.json_file
        with open(json_file) as f:
            gt = json.load(f)
        categories = gt['categories']
        file_to_id = {i['file_name']: i['id'] for i in gt['images']}
        file_to_id = {
            '/'.join(k.split('/')[-3:]): v for k, v in file_to_id.items()
        }
        id_to_annots = {id: [] for id in file_to_id.values()}
        for annot in gt['annotations']:
            annot['category'] = next(
                c['name'] for c in categories
                if c['id'] == annot['category_id']
            )
            id_to_annots[annot['image_id']].append(annot)
        return file_to_id, id_to_annots

    def _eval_loop(self, all_preds, print_every=500) -> dict:
        per_class_ap_data = {
            f: defaultdict(lambda: {'scores': [], 'labels': [], 'npos': 0})
            for f in self.ap_fields
        }
        file_to_id, id_to_annots = self._parse_data_json()

        self.print('\nStarting per-frame evaluation')
        for n, file_name in enumerate(all_preds.keys()):
            if print_every > 0 and n % print_every == 0:
                self.print('Frame: {}/{}'.format(n, len(all_preds)))

            preds = all_preds[file_name]
            try:
                annots = id_to_annots[file_to_id[file_name]]
            except KeyError:  # 400k
                img_name = file_name.split('/')[-1]
                new_img_name = '0' * (10 - len(img_name)) + img_name
                file_name = file_name.replace(img_name, new_img_name)
                annots = id_to_annots[file_to_id[file_name]]

            if not len(annots):
                for pred in preds:
                    for f in self.ap_fields:
                        ap_data = per_class_ap_data[f][pred['category']]
                        ap_data['labels'].append(0.)
                        ap_data['scores'].append(pred['score'])
                continue

            for annot in annots:
                for f in self.ap_fields:
                    per_class_ap_data[f][annot['category']]['npos'] += 1

            if not len(preds):
                continue

            preds = sorted(preds, key=lambda x: x['score'], reverse=True)

            # Box IOUs
            pred_boxes = Boxes([p['bbox'] for p in preds])
            gt_boxes = [gt['bbox'] for gt in annots]
            gt_boxes = Boxes([
                BoxMode.convert(
                    box,
                    from_mode=BoxMode.XYWH_ABS,
                    to_mode=BoxMode.XYXY_ABS
                ) for box in gt_boxes
            ])
            box_ious = pairwise_iou(pred_boxes, gt_boxes)

            # Mask IOUs
            gt_masks = mask_util.decode([gt['segmentation'] for gt in annots])
            pred_masks = mask_util.decode([p['segmentation'] for p in preds])

            pred_masks = pred_masks.reshape(-1, len(preds)).T[:, None, :]
            gt_masks = gt_masks.reshape(-1, len(annots)).T[None, :, :]

            unions = np.sum(np.logical_or(pred_masks, gt_masks), axis=-1)
            inters = np.sum(np.logical_and(pred_masks, gt_masks), axis=-1)
            mask_ious = inters / unions

            fields = ['box', 'mask']
            field_ious = [box_ious, mask_ious]

            # Collect AP labels and scores
            for field, ious in zip(fields, field_ious):
                covered = [False for _ in annots]
                for i in range(len(preds)):
                    matched = False
                    for j in range(len(annots)):
                        if covered[j]:
                            continue
                        category = preds[i]['category']
                        if category != annots[j]['category']:
                            continue
                        if ious[i, j] >= self.thresh:
                            covered[j] = True
                            matched = True
                            break
                    ap_data = per_class_ap_data[field][category]
                    ap_data['scores'].append(preds[i]['score'])
                    ap_data['labels'].append(float(matched))

        return per_class_ap_data


class DepthEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg):
        super().__init__()
        self._rendering_root = MetadataCatalog.get(dataset_name).rendering_root
        self._depth_scale = cfg.INPUT.DEPTH_SCALE
    
    def reset(self):
        self.depth_aes = []
        self.step = 0

    def process(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]]
    ):
        self.step += 1
        for input, output in zip(inputs, outputs):
            file_name = input['file_name']
            scene, _, image = file_name.split(os.sep)[-3:]
            image = image.replace('.jpg', '.png')
            depth_dir = os.path.join(
                self._rendering_root, scene, 'depth', image
            )
            gt_depth = cv.imread(depth_dir, -1)
            assert gt_depth is not None
            gt_depth = Depths.decode(gt_depth, self._depth_scale).tensor
            mask = gt_depth > 1e-5
            if not mask.any():  # Ignore empties!
                continue
            pred_depth = output['pred_image_depth'].cpu()
            mask = mask.float()
            depth_ae = masked_l1_loss(pred_depth, gt_depth, mask).item()
            self.depth_aes.append(depth_ae)

    def evaluate(self) -> OrderedDictType[str, Dict[str, float]]:
        mean_error = np.mean(self.depth_aes)
        median_error = np.median(self.depth_aes)
        mean_error, median_error = (
            np.round(100 * x, decimals=2)
            for x in (mean_error, median_error)
        )

        print('\nDepth Average Errors\n')
        print(tabulate(
            [('Mean AE', mean_error), ('Median AE', median_error)],
            tablefmt='github',
            headers=['Metric', 'Value']
        ))
        print()

        return OrderedDict({
            'depth': {'mean AE': mean_error, 'median AE': median_error}
        })
