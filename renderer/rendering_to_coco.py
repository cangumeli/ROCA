import argparse
import cv2 as cv
import json
import numpy as np
import os
import pathlib
import pycocotools.mask as mask_util
import sys
from collections import defaultdict
from tqdm import tqdm


REPS = {
    'bathtub': 4,
    'bed': 4,
    'bin': 3,
    'sofa': 3,
    'display': 3,
    'bookcase': 3,
    'cabinet': 2,
    'table': 2,
    'chair': 1
}


def annotate_roi(mask):
    rle = mask_util.encode(np.array(
        mask[:, :, None], order='F', dtype='uint8'
    ))[0]
    bbox = mask_util.toBbox(rle)
    area = mask_util.area(rle)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle, bbox.tolist(), area.tolist()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignment_json', type=str, required=True)
    parser.add_argument('--rendering_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sample_scene', type=str, default='')
    parser.add_argument('--skip_val', type=int, default=100)
    parser.add_argument('--skip_train', type=int, default=0)
    parser.add_argument('--scale_thresh', type=float, default=0)
    parser.add_argument('--benchmark_only', type=int, default=1)
    parser.add_argument('--center_filter', type=int, default=1)
    parser.add_argument('--center_filter_cat', type=int, default=0)
    parser.add_argument('--pix_area_filter', type=float, default=0)
    parser.add_argument('--cat_repeat', type=int, default=1)
    parser.add_argument('--one_file', type=int, default=1)
    parser.add_argument('--rle', type=int, default=0)
    parser.add_argument('--rle_val', type=int, default=1)
    parser.add_argument('--val_filter', type=int, default=1)
    parser.add_argument('--split', choices=['all', 'train', 'val'], default='all')
    args = parser.parse_args(args)

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    print('Loading {}...'.format(args.alignment_json))
    with open(args.alignment_json) as f:
        data = json.load(f)
        alignments = data['alignments']
        image_infos = data['images']
        config = data['config']

    path = '../metadata/scan2cad_taxonomy{}.json'.format(
        '_9' if config.get('taxonomy_9', False) else ''
    )
    with open(path) as f:
        taxonomy = json.load(f)
        s2c_names = sorted([t['name'] for t in taxonomy])
        taxonomy = {t['shapenet']: t for t in taxonomy}
        # Fix ids for deterministm
        # TODO: check other classes if not benchmark only?
        benchmark_categories = {
            name: i for i, name in enumerate(s2c_names)
        }
    del path

    with open('../metadata/labelids.txt') as f:
        benchmark_names = set(line.strip().split()[-1] for line in f)

    if args.val_filter:
        with open('../metadata/val_images.txt') as f:
            val_filter = set()
            for pair in f:
                scene, image = pair.strip().split()
                image = int(image)
                val_filter.add((scene, image))

    if config.get('all_s2c', False):
        print('All S2C mode enabled...')
        if not args.benchmark_only:
            raise RuntimeError('This mode is no longer supported!')

    train_scenes = set()
    val_scenes = set()
    for split_name, split_set in zip(('train', 'val'),
                                     (train_scenes, val_scenes)):
        with open('../metadata/scannetv2_{}.txt'.format(split_name)) as f:
            split_set.update(line.strip() for line in f)

    categories = defaultdict(lambda: len(categories), benchmark_categories)
    images = []
    annotations = []
    alignment_annotations = defaultdict(lambda: [])  # Image id -> alignments

    for k, (image_id, image_info) in tqdm(
        enumerate(image_infos.items()),
        total=len(image_infos),
        dynamic_ncols=True
    ):
        scene_id, image_name = image_id.split('/')
        '''if k % 100 == 0 or k == len(image_infos) - 1:
            print('Image {}/{}...'.format(k + 1, len(image_infos)))'''

        if args.sample_scene != '' and scene_id != args.sample_scene:
            continue
    
        if args.split == 'val' and scene_id not in val_scenes:
            continue

        if args.split =='train' and scene_id not in train_scenes:
            continue

        if (
            not args.val_filter
            and scene_id in val_scenes
            and args.skip_val > 0
            and int(image_name) % args.skip_val != 0
        ):
            # print(image_name)
            continue
        if (
            scene_id in train_scenes
            and args.skip_train > 0
            and int(image_name) % args.skip_train != 0
        ):
            continue

        if (
            args.val_filter
            and scene_id in val_scenes
            and (scene_id, int(image_name)) not in val_filter
        ):
            continue

        images.append({
            'id': len(images),
            'height': image_info['height'],
            'width': image_info['width'],
            'scene_id': scene_id,
            'file_name': 'tasks/scannet_frames_25k/{}/color/{}.jpg'.format(
                scene_id, image_name
            ),
            'duplicate': False
        })

        # print()
        instance_path = os.path.join(
            args.rendering_root,
            scene_id,
            'instance',
            image_name + '.png'
        )
        instances = cv.imread(instance_path, cv.IMREAD_GRAYSCALE)
        assert instances is not None, '{} not found!'.format(instance_path)

        image_alignments = {
            alignment['id']: alignment
            for alignment in alignments[image_id]
        }

        reps = 1
        for i in np.unique(instances):
            if i == 0:
                continue

            # Encode instance mask
            mask = instances == i
            if not np.any(mask):
                continue
            rle, bbox, area = annotate_roi(mask)
            if np.allclose(area, 0):
                continue

            if (mask.sum() / mask.size) <= args.pix_area_filter:
                continue

            # Extract class label name
            is_cad = False
            alignment = image_alignments[i]
            # import pdb; pdb.set_trace()

            if 'label_name' in alignment:  # -> ScanNet
                label_name = alignment['label_name']
                if label_name not in benchmark_names:
                    continue
                category_id = categories[label_name]

            elif 'catid_cad' in alignment:  # -> Scan2CAD
                if args.center_filter and not alignment['center_in_image']:
                    if not args.center_filter_cat:
                        continue
                    name = taxonomy.get(
                        alignment['catid_cad'], {'name': 'none'}
                    )['name']
                    if name not in ('bathtub', 'bed', 'cabinet', 'sofa'):
                    # if name not in ('bathtub', 'bed', 'table'):
                        continue

                if any(s_ < args.scale_thresh for s_ in alignment['s']):
                    continue

                try:
                    category_info = taxonomy[alignment['catid_cad']]
                    category_id = categories[category_info['name']]
                except KeyError:
                    if args.benchmark_only:
                        continue
                    category_info = shapenet_taxonomy[alignment['catid_cad']]
                    category_name = category_info['name']
                    if ',' in category_name:
                        category_name = category_name.split(',')[0]
                    category_id = categories[category_name]
                is_cad = True

                if category_info['name'] in REPS:
                    reps = max(reps, REPS[category_info['name']])

            else:  # -> Error!
                raise RuntimeError(
                    'Uncategorized alignment {}!\nImage: {}'
                    .format(alignment, image_id)
                )

            # Add coco annotation
            annotations.append({
                'id': len(annotations),
                'bbox': bbox,
                'area': area,
                'segmentation': rle,
                'iscrowd': 0,
                'image_id': images[-1]['id'],
                'category_id': category_id,
                'scene_id': scene_id,
                'is_cad': is_cad,
                'intrinsics': image_info['intrinsics'],
                'duplicate': False
            })

            # Handle rle stuff
            if (
                (not args.rle and scene_id in train_scenes) 
                or (not args.rle_val and scene_id in val_scenes)
            ):
                del annotations[-1]['segmentation']

            # Add remaining data to match nocs and instances
            mapping = {
                'bbox': annotations[-1]['bbox'],  # key
                'alignment_id': int(i),  # map to the alignment instance
                'category_id': category_id
            }
            if is_cad:
                mapping.update({
                    'q': alignment['q'],
                    's': alignment['s'],
                    't': alignment['t'],
                    'sym': alignment['sym'],
                    'model': {
                        'scene_id': scene_id,
                        'catid_cad': alignment['catid_cad'],
                        'id_cad': alignment['id_cad'],
                        'idx': alignment['id']  # For duplicate cases
                    }
                })

            if args.one_file:
                annotations[-1].update(mapping)
            else:
                alignment_annotations[images[-1]['id']].append(mapping)

        # Do repetitions of image labels
        if args.cat_repeat and scene_id in train_scenes:
            for i in range(1, reps):
                cur_id = images[-1]['id']
                new_id = len(images)
                images.append({**images[-1], 'id': new_id, 'duplicate': True})
                alignment_annotations[new_id] = alignment_annotations[cur_id]
                for annot in reversed(annotations):
                    if annot['image_id'] != cur_id:
                        break
                    annotations.append({
                        **annot,
                        'image_id': new_id,
                        'id': len(annotations),
                        'duplicate': True
                    })

    print('Writing outputs...')

    # Dump COCO instances by splitting val and train
    categories = [
        {'name': name, 'id': idx} for name, idx in categories.items()
    ]

    def select_split(data, scene_set):
        return [d for d in data if d['scene_id'] in scene_set]

    for split_name, scene_set in zip(('train', 'val'),
                                     (train_scenes, val_scenes)):

        if args.split != 'all' and split_name != args.split:
            continue

        output_data = {
            'categories': categories,
            'images': select_split(images, scene_set),
            'annotations': select_split(annotations, scene_set)
        }

        output_path = os.path.join(
            args.output_dir, 'scan2cad_instances_{}.json'.format(split_name)
        )
        with open(output_path, 'w') as f:
            json.dump(output_data, f)

    if not args.one_file:
        # Dump alignment labels
        alignment_path = os.path.join(
            args.output_dir, 'scan2cad_alignment_map.json'
        )
        with open(alignment_path, 'w') as f:
            json.dump(alignment_annotations, f)

    # Dump benchmark categories
    category_path = os.path.join(
        args.output_dir, 'scan2cad_alignment_classes.json'
    )
    with open(category_path, 'w') as f:
        json.dump(s2c_names, f)

    # Dump config
    config_path = os.path.join(
        args.output_dir, 'scan2cad_rendering_config.json'
    )
    config.update({
        'coco': vars(args)
    })
    with open(config_path, 'w') as f:
        json.dump(config, f)


if __name__ == '__main__':
    main(sys.argv[1:])
