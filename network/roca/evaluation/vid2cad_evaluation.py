import json
import os
import pickle as pkl
from collections import Counter, defaultdict, OrderedDict
from copy import deepcopy
from itertools import product
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    OrderedDict as OrderedDictType,
    Union,
)

import numpy as np
import quaternion  # noqa: F401
import torch
from pandas import DataFrame, read_csv
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Instances

from roca.data import CategoryCatalog
from roca.data.constants import (
    CAD_TAXONOMY,
    CAD_TAXONOMY_REVERSE,
    IMAGE_SIZE,
)
from roca.structures import Rotations
from roca.utils.alignment_errors import (
    rotation_diff,
    scale_ratio,
    translation_diff,
)
from roca.utils.linalg import decompose_mat4, make_M_from_tqs


NMS_TRANS = 0.4
NMS_ROT = 60
NMS_SCALE = 0.6

TRANS_THRESH = 0.2
ROT_THRESH = 20
SCALE_THRESH = 0.2
VOXEL_IOU_THRESH = 0.5


class Vid2CADEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name: str,
        full_annot: Union[str, List[Dict[str, Any]]],
        cfg=None,
        output_dir: str = '',
        mocking: bool = False,
        exclude: Optional[Iterable[str]]=None,
        grid_file: Optional[str] = None,
        exact_ret: bool = False,
        key_prefix: str = '',
        info_file: str = ''
    ):
        self._dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(self._dataset_name)
        self._category_manager = CategoryCatalog.get(self._dataset_name)

        self.mocking = mocking
        self._output_dir = output_dir

        self._exclude = exclude

        # Parse raw data
        if isinstance(full_annot, list):
            annots = full_annot
        else:
            with open(full_annot) as f:
                annots = json.load(f)
        self._full_annots = annots

        scene_alignments = {}
        scene_counts = defaultdict(lambda: Counter())
        for annot in annots:
            scene = annot['id_scan']
            trs = annot['trs']
            to_scene = np.linalg.inv(make_M_from_tqs(
                trs['translation'], trs['rotation'], trs['scale']
            ))
            alignment = []
            for model in annot['aligned_models']:
                if int(model['catid_cad']) not in CAD_TAXONOMY:
                    continue
                scene_counts[scene][int(model['catid_cad'])] += 1
                mtrs = model['trs']
                to_s2c = make_M_from_tqs(
                    mtrs['translation'],
                    mtrs['rotation'],
                    mtrs['scale']
                )
                t, q, s = decompose_mat4(to_scene @ to_s2c)
                alignment.append({
                    't': t.tolist(),
                    'q': q.tolist(),
                    's': s.tolist(),
                    'catid_cad': model['catid_cad'],
                    'id_cad': model['id_cad'],
                    'sym': model['sym']
                })
            scene_alignments[scene] = alignment

        self._scene_alignments = scene_alignments
        self._scene_counts = scene_counts

        self.with_grids = grid_file is not None
        self.grid_data = None
        if self.with_grids:
            with open(grid_file, 'rb') as f:
                self.grid_data = pkl.load(f)

        self.exact_ret = exact_ret
        self.key_prefix = key_prefix
        self.info_file = info_file

    def reset(self):
        self.results = defaultdict(list)
        self.poses = defaultdict(list)
        self.object_ids = defaultdict(list)
        self.info_data = defaultdict(list)

    def process(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]]
    ):
        for input, output in zip(inputs, outputs):
            file_name = input['file_name']
            scene_name = input['file_name'].split('/')[-3]

            if 'instances' not in output:
                continue
            instances = output['instances']
            instances = instances[instances.scores > 0.5]
            has_alignment = instances.has_alignment
            if not has_alignment.any():
                continue
            instances = instances[has_alignment]
            instances = deepcopy(instances)  # avoid modification!
            if instances.has('pred_meshes'):
                instances.remove('pred_meshes')
            self.results[scene_name].append(instances.to('cpu'))

            if 'cad_ids' in output:
                object_ids = output['cad_ids']
                object_ids = [
                    object_ids[i]
                    for i in instances.pred_indices.tolist()
                ]
                self.object_ids[scene_name].append(object_ids)

            pose_file = file_name\
                .replace('color', 'pose')\
                .replace('.jpg', '.txt')
            with open(pose_file) as f:
                pose_mat = torch.tensor([
                    [float(v) for v in line.strip().split()]
                    for line in f
                ])
            pose_mat = pose_mat.unsqueeze(0).expand(len(instances), 4, 4)
            self.poses[scene_name].append(pose_mat)
        '''if len(self.poses) == 20:
            self.evaluate()

            exit(0)'''

    def process_mock(
        self,
        scene_name: str,
        instances: Instances,
        object_ids=None
    ):
        self.results[scene_name] = instances
        self.object_ids[scene_name] = object_ids

    def evaluate(self) -> OrderedDictType[str, Dict[str, float]]:
        self._collect_results()
        self._transform_results_to_world_space()
        path = self._write_raw_results()
        return eval_csv(
            self._dataset_name,
            path,
            self._full_annots,
            exact_ret=self.exact_ret,
            prefix=self.key_prefix,
            info_file=self.info_file
        )

    def evaluate_mock(self) -> OrderedDictType[str, Dict[str, float]]:
        self._nms_results()
        self._apply_constraints()
        return self._compute_metrics()

    def _collect_results(self):
        print('INFO: Collecting results...', flush=True)

        for k, v in self.results.items():
            instances = Instances.cat(v)
            indices = instances.scores.argsort(descending=True)
            self.results[k] = instances[indices]
            self.poses[k] = torch.cat(self.poses[k], dim=0)[indices]

            # NOTE: Objects corresponds to instances,
            # so sort them similar to results
            if k in self.object_ids:
                object_ids = []
                for ids in self.object_ids[k]:
                    object_ids.extend(ids)
                self.object_ids[k] = [object_ids[i] for i in indices.tolist()]

    def _transform_results_to_world_space(self):
        print('INFO: Transforming results to world space...', flush=True)

        for scene, instances in self.results.items():
            poses = self.poses[scene]

            # TODO: This can be batched
            for i, (pose, t, q, s) in enumerate(zip(
                poses.unbind(0),
                instances.pred_translations.unbind(0),
                instances.pred_rotations.unbind(0),
                instances.pred_scales.unbind(0)
            )):
                pose = pose.numpy().reshape(4, 4)
                mat = make_M_from_tqs(t.tolist(), q.tolist(), s.tolist())
                new_t, new_q, new_s = decompose_mat4(pose @ mat)

                instances.pred_translations[i] = torch.from_numpy(new_t)
                instances.pred_rotations[i] = torch.from_numpy(new_q)
                instances.pred_scales[i] = torch.from_numpy(new_s)

            self.results[scene] = instances
            self.poses[scene] = poses

    def _write_raw_results(self):
        output_dir = self._output_dir
        output_path = os.path.join(output_dir, 'raw_results.csv')
        print(
            'INFO: Writing raw results to {}...'.format(output_path),
            flush=True
        )

        data = defaultdict(lambda: [])
        results = sorted(self.results.items(), key=lambda x: x[0])
        for scene, instances in results:
            data['id_scan'].extend((scene,) * len(instances))
            for c in instances.pred_classes.tolist():
                cid = CAD_TAXONOMY_REVERSE[self._category_manager.get_name(c)]
                data['objectCategory'].append(cid)

            data['alignedModelId'].extend(
                id_cad for _, id_cad in self.object_ids[scene]
            )

            data['tx'].extend(instances.pred_translations[:, 0].tolist())
            data['ty'].extend(instances.pred_translations[:, 1].tolist())
            data['tz'].extend(instances.pred_translations[:, 2].tolist())

            data['qw'].extend(instances.pred_rotations[:, 0].tolist())
            data['qx'].extend(instances.pred_rotations[:, 1].tolist())
            data['qy'].extend(instances.pred_rotations[:, 2].tolist())
            data['qz'].extend(instances.pred_rotations[:, 3].tolist())

            data['sx'].extend(instances.pred_scales[:, 0].tolist())
            data['sy'].extend(instances.pred_scales[:, 1].tolist())
            data['sz'].extend(instances.pred_scales[:, 2].tolist())

            data['object_score'].extend(instances.scores.tolist())

        frame = DataFrame(data=data)
        frame.to_csv(output_path, index=False)
        return output_path

    def _nms_results(self):
        print('INFO: Removing duplicate results...', flush=True)

        for scene, instances in self.results.items():
            pred_trans = instances.pred_translations
            pred_rot = Rotations(instances.pred_rotations).as_quaternions()
            pred_scale = instances.pred_scales
            pred_classes = instances.pred_classes
            num_instances = len(instances)
            # scores = instances.scores

            all_pairs = product(
                range(num_instances), reversed(range(num_instances))
            )
            valid_map = torch.ones(len(instances), dtype=torch.bool)
            for i, j in all_pairs:
                '''if scores[j] >= scores[i]:
                    continue'''
                # NOTE: Assume sorted by score
                if i >= j:
                    continue
                # NOTE: if the high score one was removed earlier,
                # do not drop the low score one
                if not valid_map[i] or not valid_map[j]:
                    continue
                if pred_classes[j] != pred_classes[i]:
                    continue

                object_ids = self.object_ids[scene]
                if self.mocking:
                    cat_i = pred_classes[i]
                    model_i = object_ids[instances.model_indices[i].item()]
                else:
                    cat_i, model_i = object_ids[i]
                sym = next(
                    a['sym']
                    for a in self._scene_alignments[scene]
                    if int(a['catid_cad']) == int(cat_i)
                    and a['id_cad'] == model_i
                )

                is_dup = (
                    translation_diff(pred_trans[i], pred_trans[j]) <= NMS_TRANS
                    and scale_ratio(pred_scale[i], pred_scale[j]) <= NMS_SCALE
                    and rotation_diff(pred_rot[i], pred_rot[j], sym) <= NMS_ROT
                )
                if is_dup:
                    valid_map[j] = False

            self.results[scene] = instances[valid_map]
    
    def _apply_constraints(self):
        print('INFO: Applying Scan2CAD constraints...', flush=True)
        for scene, instances in self.results.items():
            gt_counts = self._scene_counts[scene]
            pred_counts = Counter()
            mask = torch.ones(len(instances), dtype=torch.bool)
            for i, catid in enumerate(instances.pred_classes.tolist()):
                if pred_counts[catid] >= gt_counts[catid]:
                    mask[i] = False
                else:
                    pred_counts[catid] += 1
            self.results[scene] = instances[mask]

    def _compute_metrics(self):
        print('INFO: Computing final metrics...', flush=True)

        corrects_per_class = Counter()
        counts_per_class = Counter()
        for scene, instances in self.results.items():
            corrects, counts = self._count_corrects(scene, instances)
            corrects_per_class.update(corrects)
            counts_per_class.update(counts)

        if self.info_file != '':
            print('Writing evaluation info to {}...'.format(self.info_file))
            with open(self.info_file, 'w') as f:
                json.dump(dict(self.info_data), f)

        if self._exclude is not None:
            for scene in self._exclude:
                for cat, count in self._scene_counts[scene].items():
                    if cat in counts_per_class:
                        counts_per_class[cat] += count

        if not self.mocking:
            corrects_per_class = Counter({
                self._category_manager.get_name(k): v
                for k, v in corrects_per_class.items()
            })
            counts_per_class = Counter({
                self._category_manager.get_name(k): v
                for k, v in counts_per_class.items()
            })
        else:
            corrects_per_class = Counter({
                CAD_TAXONOMY[k]: v
                for k, v in corrects_per_class.items()
            })
            counts_per_class = Counter({
                CAD_TAXONOMY[k]: v
                for k, v in counts_per_class.items()
            })

        accuracies = OrderedDict()
        # import pdb; pdb.set_trace()
        for cat in counts_per_class.keys():
            accuracies[cat] = np.round(
                100 * corrects_per_class[cat] / counts_per_class[cat],
                decimals=1
            )

        print()
        print(tabulate(
            sorted(accuracies.items(), key=lambda x: x[0]),
            tablefmt='github',
            headers=['class', 'accuracy']
        ))

        category_average = np.mean(list(accuracies.values()))
        benchmark_average = np.mean([
            acc for cat, acc in accuracies.items()
            if self._category_manager.is_benchmark_class(cat)
        ])
        instance_average = 100 * (
            sum(corrects_per_class.values()) / sum(counts_per_class.values())
        )

        instance_benchmark_average = 100 * (
            sum(
                val for cat, val in corrects_per_class.items()
                if self._category_manager.is_benchmark_class(cat)
            ) / sum(
                val for cat, val in counts_per_class.items()
                if self._category_manager.is_benchmark_class(cat)
            )
        )

        metrics = OrderedDict({
            'category': np.round(category_average, decimals=1),
            'benchmark': np.round(benchmark_average, decimals=1),
            'instance (all)': np.round(instance_average, decimals=1),
            'instance (benchmark)':
                np.round(instance_benchmark_average, decimals=1)
        })
        print()
        print(tabulate(
            list(metrics.items()),
            tablefmt='github',
            headers=['metric', 'accuracy']
        ))
        print()

        return OrderedDict({self.key_prefix + 'alignment': metrics})

    def _count_corrects(self, scene, instances):
        labels = self._scene_alignments[scene]
        if not self.mocking:
            class_map = self._metadata.thing_dataset_id_to_contiguous_id
            labels = [
                {**label, 'category_id': class_map[label['category_id']]}
                for label in labels
            ]

        label_counts = Counter()
        for label in labels:
            if not self.mocking:
                label_counts[label['category_id']] += 1
            else:
                label_counts[int(label['catid_cad'])] += 1

        corrects = Counter()
        covered = [False for _ in labels]
        for i in range(len(instances)):
            pred_trans = instances.pred_translations[i]
            pred_rot = np.quaternion(*instances.pred_rotations[i].tolist())
            pred_scale = instances.pred_scales[i]
            pred_class = instances.pred_classes[i].item()

            object_ids = self.object_ids[scene]
            if self.mocking:
                model_i = object_ids[instances.model_indices[i].item()]
                cat_i = pred_class
            else:
                cat_i, model_i = object_ids[i]
            sym_i = next(
                a['sym']
                for a in self._scene_alignments[scene]
                if int(a['catid_cad']) == int(cat_i)
                and a['id_cad'] == model_i
            )

            match = None
            for j, label in enumerate(labels):
                if covered[j]:
                    continue
                gt_class = (
                    label['category_id']
                    if not self.mocking
                    else label['catid_cad']
                )
                if pred_class != int(gt_class):
                    continue

                gt_trans = torch.tensor(label['t'])
                gt_rot = np.quaternion(*label['q'])
                gt_scale = torch.tensor(label['s'])
                if sym_i == label['sym']:
                    angle_diff = rotation_diff(pred_rot, gt_rot, sym_i)
                else:
                    angle_diff = rotation_diff(pred_rot, gt_rot)
                is_correct = (
                    translation_diff(pred_trans, gt_trans) <= TRANS_THRESH
                    and angle_diff <= ROT_THRESH
                    and scale_ratio(pred_scale, gt_scale) <= SCALE_THRESH
                )

                if self.exact_ret:
                    cad_pred = (cat_i, model_i)
                    cad_gt = (int(label['catid_cad']), label['id_cad'])
                    is_correct = is_correct and cad_pred == cad_gt
                elif self.with_grids:
                    try:
                        iou = self._voxel_iou(cat_i, model_i, label)
                        # print('Passed')
                    except KeyError:
                        iou = 1.0
                        print('failed')
                    is_correct = is_correct and iou >= VOXEL_IOU_THRESH

                if is_correct:
                    corrects[pred_class] += 1
                    covered[j] = True
                    match = {'index': j, 'label': label}
                    break

            if self.info_file != '':
                self.info_data[scene].append({
                    'id_cad': model_i,
                    'catid_cad': cat_i,
                    'match': match,
                    't': pred_trans.tolist(),
                    'q': quaternion.as_float_array(pred_rot).tolist(),
                    's': pred_scale.tolist()
                })

        return corrects, label_counts

    def _voxel_iou(self, cat_i, model_i, label):
         # import pdb; pdb.set_trace()
        pred_ind = self.grid_data['0' + str(cat_i), model_i]
        gt_ind = \
            self.grid_data[label['catid_cad'], label['id_cad']]
        
        pred_ind_1d = np.ravel_multi_index(
            multi_index=(pred_ind[:, 0], pred_ind[:, 1], pred_ind[:, 2]),
            dims=(32, 32, 32)
        )
        gt_ind_1d = np.ravel_multi_index(
            multi_index=(gt_ind[:, 0], gt_ind[:, 1], gt_ind[:, 2]),
            dims=(32, 32, 32)
        )

        inter = np.intersect1d(pred_ind_1d, gt_ind_1d).size
        union = pred_ind_1d.size + gt_ind_1d.size - inter
        return inter / union


def eval_csv(
    dataset_name: str,
    csv_path: str,
    full_annot=None,
    grid_file=None,
    exact_ret=False,
    prefix: str = '',
    info_file=''
) -> OrderedDictType[str, Dict[str, float]]:

    # FIXME: relative path!
    eval_path = __file__
    for i in range(4):  # -> eval -> roca -> network
        eval_path = os.path.dirname(eval_path)
    eval_path = os.path.join(eval_path, 'metadata', 'scannetv2_val.txt')
    with open(eval_path) as f:
        val_scenes = set(ln.strip() for ln in f)

    data = read_csv(csv_path)
    scenes = data['id_scan'].unique()
    exclude = val_scenes.difference(scenes)
    evaluator = Vid2CADEvaluator(
        dataset_name,
        full_annot=full_annot,
        mocking=True,
        exclude=exclude,
        grid_file=grid_file,
        exact_ret=exact_ret,
        key_prefix=prefix,
        info_file=info_file
    )
    evaluator.reset()

    print('INFO: Processing outputs...')
    for i, scene in enumerate(scenes):
        # print('{} / {}'.format(i, len(scenes)))
        scene_data = data[data['id_scan'] == scene]

        pred_trans = np.hstack([
            scene_data['tx'][:, None],
            scene_data['ty'][:, None],
            scene_data['tz'][:, None]
        ])
        pred_rot = np.hstack([
            scene_data['qw'][:, None],
            scene_data['qx'][:, None],
            scene_data['qy'][:, None],
            scene_data['qz'][:, None]
        ])
        pred_scale = np.hstack([
            scene_data['sx'][:, None],
            scene_data['sy'][:, None],
            scene_data['sz'][:, None]
        ])
        pred_catids = np.asarray(scene_data['objectCategory']).tolist()
        scores = np.asarray(scene_data['object_score'])

        model_list = scene_data['alignedModelId'].tolist()
        model_indices = torch.arange(len(model_list), dtype=torch.long)

        instances = Instances(IMAGE_SIZE)
        instances.pred_translations = torch.from_numpy(pred_trans).float()
        instances.pred_rotations = torch.from_numpy(pred_rot).float()
        instances.pred_scales = torch.from_numpy(pred_scale).float()
        instances.pred_classes = torch.tensor(pred_catids)
        instances.scores = torch.from_numpy(scores).float()
        instances.model_indices = model_indices
        instances = instances[instances.scores.argsort(descending=True)]

        evaluator.process_mock(scene, instances, model_list)

    return evaluator.evaluate_mock()
