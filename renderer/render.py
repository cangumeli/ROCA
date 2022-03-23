import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import pathlib
import pickle
import sys
import time
import torch
from collections import defaultdict
from queue import Empty, Queue

from utils.io import (
    load_image_size,
    load_intrinsics,
    load_mesh,
    load_poses,
    write_images,
)
from utils.labeling import (
    AreaFilter,
    LabelFilter,
    LabelMerger,
)
from utils.linalg import (
    decompose_mat4,
    from_hom,
    make_M_from_tqs,
    to_hom,
)
from utils.raster import Rasterizer


def load_models(d: dict, cad_root: str):
    trs = d['trs']
    scan_to_s2c = make_M_from_tqs(
        trs['translation'],
        trs['rotation'],
        trs['scale']
    )
    s2c_to_scan = np.linalg.inv(scan_to_s2c)

    models = d['aligned_models']
    meshes = []
    model_to_scans = []
    cache = {}

    for model in models:
        cat_id = model['catid_cad']
        cad_id = model['id_cad']

        obj_path = os.path.abspath(
            '{}/{}/{}/models/model_normalized.obj'.
            format(cad_root, cat_id, cad_id)
        )
        if obj_path in cache:
            mesh = cache[obj_path]
            mesh = mesh.clone()  # Copy
        else:
            mesh = load_mesh(obj_path)
            cache[obj_path] = mesh.clone()

        model_trs = model['trs']
        model_to_s2c = make_M_from_tqs(
            model_trs['translation'],
            model_trs['rotation'],
            model_trs['scale']
        )

        model_to_scan = s2c_to_scan @ model_to_s2c
        model_to_scans.append(model_to_scan)

        # mesh.transform(model_to_scan)
        points = mesh.verts_list()[0]
        points = torch.from_numpy(
            from_hom(
                to_hom(points.numpy()) @ model_to_scan.T
            )
        )
        mesh = type(mesh)(verts=[points], faces=mesh.faces_list())

        meshes.append(mesh)

    return meshes, models, model_to_scans


def render_scene(
    args: argparse.Namespace,
    meshes: list,
    models: list,
    model_to_scans: list,
    scene_id: str,
    taxonomy: list,
    label_ids: list,
    matches: dict
):
    alignments = defaultdict(lambda: [])
    images = {}

    scene_dir = os.path.join(args.scan_root, scene_id)

    vertices = [mesh.verts_list()[0] for mesh in meshes]
    mesh_centers = [
        to_hom(.5 * (vert.max(0)[0] + vert.min(0)[0]).numpy())
        for vert in vertices
    ]
    del vertices

    poses = load_poses(scene_dir)
    intrinsics = load_intrinsics(scene_dir)
    height, width = load_image_size(scene_dir)

    categories = set(t['shapenet'] for t in taxonomy)

    rasterizer = Rasterizer(width, height, intrinsics)
    area_filter = AreaFilter(ratio=args.min_ratio)
    label_filter = None
    if not args.label_merge_2d:
        label_filter = LabelFilter(
            scene_dir=scene_dir,
            match_ids=matches['to_remove'][scene_id],
            num_models=len(meshes),
            taxonomy=taxonomy,
            label_ids=label_ids,
            label_names=matches['id_to_nyu'][scene_id]
        )

    for name, pose in sorted(poses.items(), key=lambda pair: int(pair[0])):
        # Reset the rendering process
        rasterizer.clear_models()

        if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
            print('skipping {}: pose is {}'.format(name, pose))
            continue

        scan_to_camera = np.linalg.inv(pose)

        alignment_key = '{}/{}'.format(scene_id, name)

        for idx, (model_to_scan, mesh, model, center) in enumerate(zip(
            model_to_scans,
            meshes,
            models,
            mesh_centers
        )):
            # Reserve zero to background
            idx = idx + 1

            '''# Filter out objects whose centers are outside the image
            center_in_image = True
            center = intrinsics @ scan_to_camera @ center
            x, y, z = from_hom(center).tolist()
            if z <= 0:
                center_in_image = False
            else:
                row, col = y / z, x / z
                in_image = ((0 <= row < height) and (0 <= col < width))
                if not in_image:
                    center_in_image = False
            '''

            # Filter out non-benchmark classes
            if not args.all_s2c and model['catid_cad'] not in categories:
                continue

            # Transform the mesh
            # mesh = type(mesh)(mesh)  # copy
            # mesh.transform(scan_to_camera)
            point_list = mesh.verts_list()
            assert len(point_list) == 1
            points = from_hom(to_hom(point_list[0].numpy()) @ scan_to_camera.T)
            mesh = type(mesh)(
                verts=[torch.from_numpy(points).float()],
                faces=mesh.faces_list()
            )

            model_to_camera = scan_to_camera @ model_to_scan
            camera_to_model = np.linalg.inv(model_to_camera)

            rasterizer.add_mesh(mesh, idx, camera_to_model)

            t, q, s = decompose_mat4(model_to_camera)

             # Apply center filter in camera space
            center_in_image = True
            x, y, z = from_hom(intrinsics @ to_hom(t)).tolist()
            if z <= 0:
                center_in_image = False
            else:
                row, col = y / z, x / z
                center_in_image = (0 <= row < height) and (0 <= col < width)

            alignments[alignment_key].append({
                't': t.tolist(),
                'q': q.tolist(),
                's': s.tolist(),
                'catid_cad': model['catid_cad'],
                'id_cad': model['id_cad'],
                'id': idx,
                'center_in_image': center_in_image,
                'sym': model['sym']
            })

        rasterizer.rasterize()

        try:
            # print(name)
            if args.all_s2c:
                instance_image = rasterizer.read_idx()
                new_instances = []
            elif args.label_merge_2d:
                instance_image, new_instances = LabelMerger(
                    scene_dir=scene_dir,
                    image_name=name,
                    taxonomy=taxonomy,
                    label_ids=label_ids,
                    idx=rasterizer.read_idx(),
                    alignments=alignments[alignment_key],
                    num_models=len(meshes)
                ).result
            else:
                instance_image, new_instances = label_filter(
                    idx=rasterizer.read_idx(),
                    image_name=name,
                    alignments=alignments[alignment_key]
                )

        except Exception as e:
            print('Skipping {}/{} due to an exception {}'.format(
                scene_id, name, str(e)
            ))
            continue

        alignments[alignment_key].extend(new_instances)
        instance_image = area_filter(instance_image)
        instance_mask = (instance_image != 0)
        noc = (
            rasterizer.read_noc() * instance_mask[:, :, None]
            if args.render_nocs
            else None
        )

        write_images(
            output_scene_dir=os.path.join(args.output_image_dir, scene_id),
            output_image_name=name,
            noc=noc,
            depth=rasterizer.read_depth() * instance_mask,
            idx=instance_image,
            noc_offset=args.noc_offset,
            noc_scale=args.noc_scale,
            depth_scale=args.depth_scale
        )

        images[alignment_key] = {
            'height': height,
            'width': width,
            'intrinsics': intrinsics[:3, :3].tolist()
        }

    return dict(alignments), images


def worker(
    args: argparse.Namespace,
    data_queue: mp.Queue,
    taxonomy,
    label_ids,
    matches,
    process_idx,
    lock=None
):
    pid = os.getpid()
    alignments = {}
    images = {}

    while True:
        try:
            # datum = data_queue.get_nowait()
            datum = data_queue.get(timeout=20)
        except Empty:
            break

        if lock is not None:
            lock.acquire()
        print('Worker {} with pid {} is processing {} ({} / {})'.format(
            process_idx, pid, datum['id_scan'],
            datum['index'], datum['total_size']
        ))
        if lock is not None:
            lock.release()

        scene_id = datum['id_scan']

        meshes, models, model_to_scans = load_models(datum, args.cad_root)
        scene_alignments, scene_images = render_scene(
            args,
            meshes,
            models,
            model_to_scans,
            scene_id,
            taxonomy,
            label_ids,
            matches
        )
        alignments.update(scene_alignments)
        images.update(scene_images)

    output_data = {
        'alignments': alignments,
        'images': images
    }

    # output_data is too big for a queue, this solution is not optimal but safe
    if process_idx > 0:
        with open('{}.s2c.local'.format(process_idx), 'wb') as f:
            pickle.dump(output_data, f)
    else:
        return output_data


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2c_root', type=str, required=True)
    parser.add_argument('--cad_root', type=str, required=True)
    parser.add_argument('--scan_root', type=str, required=True)
    parser.add_argument('--output_json_dir', type=str, required=True)
    parser.add_argument('--output_image_dir', type=str, required=True)
    parser.add_argument('--match_json', type=str, default='')

    parser.add_argument('--noc_offset', type=float, default=1.)
    parser.add_argument('--noc_scale', type=int, default=10000)
    parser.add_argument('--depth_scale', type=int, default=1000)
    parser.add_argument('--min_ratio', type=float, default=0)
    parser.add_argument('--label_merge_2d', type=int, default=1)
    parser.add_argument('--render_nocs', type=int, default=0)
    parser.add_argument('--all_s2c', type=int, default=1)
    parser.add_argument('--taxonomy_9', type=int, default=1)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--skip', type=int, default=-1)
    parser.add_argument('--start_file', type=str, default='')

    args = parser.parse_args(args)
    print(args)

    pathlib.Path(args.output_image_dir).mkdir(exist_ok=True, parents=True)
    pathlib.Path(args.output_json_dir).mkdir(exist_ok=True, parents=True)

    images = {}
    alignments = {}
    if args.start_file != '':
        with open(args.start_file) as f:
            start = json.load(f)
            images = start['images']
            alignments = start['alignments']

    assert args.match_json != '' or args.label_merge_2d, \
        'Argparse: You must provide match_json file or set label_merge_2d to 1'

    def clean_up():
        for fname in os.listdir(os.getcwd()):
            if '.s2c.local' in fname:
                os.unlink(os.path.join(os.getcwd(), fname))

    clean_up()

    with open(os.path.join(args.s2c_root, 'full_annotations.json')) as f:
        data = json.load(f)
        if args.skip > 0:
            data = data[args.skip:]
        if args.limit > 0:
            limit = min(args.limit, len(data))
            data = data[:limit]
        used_ids = set(k.split('/')[0] for k in images.keys())
        data = [d for d in data if d['id_scan'] not in used_ids]

    path = '../metadata/scan2cad_taxonomy{}.json'.format(
        '_9' if args.taxonomy_9 else ''
    )
    print('Reading {}...'.format(path))
    with open(os.path.abspath(path)) as f:
        taxonomy = json.load(f)
    del path

    with open(os.path.abspath('../metadata/labelids_all.txt')) as f:
        label_ids = []
        for line in f:
            parts = line.split()
            label_ids.append((int(parts[0]), ' '.join(parts[1:])))

    matches = None
    if not args.label_merge_2d:
        with open(args.match_json) as f:
            matches = json.load(f)

    distributed = args.num_workers > 0

    if distributed:
        data_queue = mp.Queue(maxsize=len(data))
        lock = mp.Lock()
        workers = [
            mp.Process(
                target=worker,
                args=(
                    args,
                    data_queue,
                    taxonomy,
                    label_ids,
                    matches,
                    i + 1, 
                    lock,
                )
            ) for i in range(args.num_workers)
        ]
    else:
        data_queue = Queue(maxsize=len(data))

    for i, d in enumerate(data):
        d['index'] = i
        d['total_size'] = len(data)
        scene_id = d['id_scan']
        # if scene_id not in ('scene0697_01', 'scene0356_02', 'scene0678_01'):
        #    continue
        if scene_id != 'scene0470_00':
            continue
        data_queue.put(d)

    if distributed:
        for process in workers:
            process.start()

        for process in workers:
            process.join()
    else:
        local_data = worker(args, data_queue, taxonomy, label_ids, matches, 0)

    print('Collecting data...')

    def update(local_data):
        images.update(local_data['images'])
        alignments.update(local_data['alignments'])

    if distributed:
        for fname in os.listdir(os.getcwd()):
            if '.s2c.local' in fname:
                path = os.path.join(os.getcwd(), fname)
                with open(path, 'rb') as f:
                    local_data = pickle.load(f)
                del path
                update(local_data)
    else:
        update(local_data)

    output_data = {
        'alignments': alignments,
        'images': images,
        'config': {
            'all_s2c': args.all_s2c,
            'noc_scale': args.noc_scale,
            'noc_offset': args.noc_offset,
            'depth_scale': args.depth_scale,
            'min_ratio': args.min_ratio,
            'taxonomy_9': args.taxonomy_9
        }
    }
    output_path = os.path.join(
        args.output_json_dir, 'scan2cad_image_alignments.json'
    )
    print('Writing alignments to {}...'.format(output_path))
    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    print('Cleaning up...')
    clean_up()


if __name__ == '__main__':
    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))
