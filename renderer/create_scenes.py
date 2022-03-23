import argparse
import json
import numpy as np
import os
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path
from pytorch3d.io import save_ply

from utils.io import load_mesh
from utils.linalg import decompose_mat4, make_M_from_tqs, transform_mesh


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2c_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--export_meshes', type=int, default=0)
    parser.add_argument('--cad_root', type=str, default='none')
    parser.add_argument('--filter_vis', type=int, default=0)
    args = parser.parse_args(args)

    if args.export_meshes:
        assert args.cad_root != 'none', 'CAD root is needed for mesh export'

    s2c_path = os.path.join(args.s2c_dir, 'full_annotations.json')
    with open(s2c_path) as f:
        s2c_data = json.load(f)

    try:
        val_path = os.path.join(args.data_dir, 'scan2cad_alignment_map.json')
        with open(val_path) as f:
            image_data = chain(*json.load(f).values())
    except FileNotFoundError:
        print('One file assumed')
        val_path = os.path.join(args.data_dir, 'scan2cad_instances_val.json')
        with open(val_path) as f:
            image_data = json.load(f)['annotations']

    # Collect visible instances
    instances_by_scene = defaultdict(lambda: set())
    category_by_key = {}
    for instance in image_data:
        if 'model' in instance:
            model_info = instance['model']
            key = (
                model_info['idx'] - 1,
                model_info['catid_cad'],
                model_info['id_cad'],
            )
            instances_by_scene[model_info['scene_id']].add(key)
            category_by_key[key] = instance['category_id']

    with open('../metadata/scannetv2_val.txt') as f:
        val_scenes = set(line.strip() for line in f)

    # Create scenes with CAD models in the world space
    cads_by_scene = {}
    for scene_data in s2c_data:
        scene_id = scene_data['id_scan']
        if scene_id not in val_scenes:
            continue
        print('{} / {}'.format(len(cads_by_scene), len(val_scenes)))

        # Collect and transform visible CAD ids
        trs = scene_data['trs']
        to_scene = np.linalg.inv(make_M_from_tqs(
            trs['translation'],
            trs['rotation'],
            trs['scale']
        ))

        instances = instances_by_scene[scene_id]
        models = []
        for i, model in enumerate(scene_data['aligned_models']):
            key = (i, model['catid_cad'], model['id_cad'])
            if args.filter_vis and key not in instances:
                continue
            to_s2c = make_M_from_tqs(
                model['trs']['translation'],
                model['trs']['rotation'],
                model['trs']['scale']
            )
            to_world = to_scene @ to_s2c
            t, q, s = decompose_mat4(to_world)

            if key in category_by_key:  # CAD is visible
                category = category_by_key[key]
            else:
                # Pick the first category with same catid
                try:
                    category = next(
                        category_by_key[key_]
                        for key_ in category_by_key.keys()
                        if key[1] == key_[1]
                    )
                    # import pdb; pdb.set_trace()
                except StopIteration:  # Non-benchmark category
                    continue
            # import pdb; pdb.set_trace()
            models.append({
                'idx': i + 1,
                'catid_cad': model['catid_cad'],
                'id_cad': model['id_cad'],
                't': t.tolist(),
                'q': q.tolist(),
                's': s.tolist(),
                'to_scene': to_scene.tolist(),
                'category_id': category,
                'sym': model['sym']
            })

            # Export mesh if needed
            # FIXME: duplicate mesh loading possibility
            if args.export_meshes:
                dir = os.path.join(args.data_dir, 'meshes', scene_id)
                path = os.path.join(dir, '{}_{}_{}.ply'.format(
                    model['catid_cad'], model['id_cad'], i+1
                ))
                Path(dir).mkdir(exist_ok=True, parents=True)
                cad_path = os.path.join(
                    args.cad_root,
                    model['catid_cad'],
                    model['id_cad'],
                    'models',
                    'model_normalized.obj'
                )
                mesh = load_mesh(cad_path)
                mesh = transform_mesh(mesh, to_world)
                save_ply(path, mesh.verts_list()[0], mesh.faces_list()[0])

        cads_by_scene[scene_id] = models

    output_path = os.path.join(args.data_dir, 'scan2cad_val_scenes.json')
    with open(output_path, 'w') as f:
        json.dump(cads_by_scene, f)


if __name__ == '__main__':
    main(sys.argv[1:])
