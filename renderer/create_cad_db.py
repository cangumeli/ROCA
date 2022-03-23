import argparse
import json
import os
import pickle
import sys
from itertools import chain

from utils.io import load_mesh


def load_meshes(args: argparse.Namespace, mappings: list):
    meshes = []
    unique_cads = set(
        (model['catid_cad'], model['id_cad'],
         mapping['category_id'], mapping['sym'])
        for model, mapping in zip(
            map(lambda m: m['model'], mappings),
            mappings
        )
    )
    num_cads = len(unique_cads)
    print('Loading {} unique CADs...'.format(num_cads))
    for i, cad_id in enumerate(unique_cads):
        if i % 100 == 0:
            print('{} / {}'.format(i, num_cads))
        cad_id, category_id, sym = cad_id[:2], cad_id[-2], cad_id[-1]
        cad_path = os.path.join(
            args.cad_root,
            *cad_id,
            'models',
            'model_normalized.obj'
        )
        verts, faces = load_mesh(cad_path, as_numpy_tuple=True)
        meshes.append({
            'verts': verts,
            'faces': faces,
            'catid_cad': cad_id[0],
            'id_cad': cad_id[1],
            'category_id': category_id,
            'sym': sym
        })
    return meshes


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cad_root', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args(args)

    with open('../metadata/scannetv2_train.txt') as f:
        train_scenes = set(line.strip() for line in f)
    with open('../metadata/scannetv2_val.txt') as f:
        val_scenes = set(line.strip() for line in f)

    try:
        mapping_file = os.path.join(args.data_dir, 'scan2cad_alignment_map.json')
        print('Loading the alignment map {}...'.format(mapping_file))
        with open(mapping_file) as f:
            mappings = json.load(f)

        train_mappings, val_mappings = [], []
        for mapping in filter(
            lambda m: 'model' in m, chain(*mappings.values())
        ):
            scene_id = mapping['model']['scene_id']
            if scene_id in train_scenes:
                train_mappings.append(mapping)
            elif scene_id in val_scenes:
                val_mappings.append(mapping)
            else:
                print('Out of split scene {}, please check!'.format(scene_id))
    except FileNotFoundError:  # one file
        print('Assumed one file')
        train_file = os.path.join(
            args.data_dir, 'scan2cad_instances_train.json'
        )
        val_file = os.path.join(
            args.data_dir, 'scan2cad_instances_val.json'
        )
        with open(train_file) as f:
            train_data = json.load(f)
        with open(val_file) as f:
            val_data = json.load(f)
        train_mappings = train_data['annotations']
        val_mappings = val_data['annotations']

    print('Loading training meshes...')
    train_meshes = load_meshes(args, train_mappings)
    print('Loading val meshes...')
    val_meshes = load_meshes(args, val_mappings)

    train_path = os.path.join(args.data_dir, 'scan2cad_train_cads.pkl')
    val_path = os.path.join(args.data_dir, 'scan2cad_val_cads.pkl')

    print('Writing training meshes to {}...'.format(train_path))
    with open(train_path, 'wb') as f:
        pickle.dump(train_meshes, f)

    print('Writing val meshes to {}...'.format(val_path))
    with open(val_path, 'wb') as f:
        pickle.dump(val_meshes, f)


if __name__ == '__main__':
    main(sys.argv[1:])
