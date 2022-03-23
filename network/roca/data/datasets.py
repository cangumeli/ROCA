import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from roca.data.cad_manager import register_cads
from roca.data.category_manager import register_categories


def register_scan2cad(
    name: str,
    metadata: dict,
    full_annot: str,
    data_dir: str,
    image_root: str,
    rendering_root: str,
    split: str,
    class_freq_method: str = 'none'
):
    json_file = os.path.join(
        data_dir, 'scan2cad_instances_{}.json'.format(split)
    )
    cad_file = os.path.join(data_dir, 'scan2cad_{}_cads.pkl'.format(split))
    category_file = os.path.join(data_dir, 'scan2cad_alignment_classes.json')
    point_file = os.path.join(data_dir, 'points_{}.pkl'.format(split))
    if not os.path.isfile(point_file):
        point_file = None if split == 'train' else 'assets/points_val.pkl'
    grid_file = os.path.join(data_dir, '{}_grids_32.pkl'.format(split))

    # TODO: may need a train scenes in the future
    scene_file = None
    if split == 'val':
        scene_file = os.path.join(data_dir, 'scan2cad_val_scenes.json')

    extra_keys = ['t', 'q', 's', 'intrinsics', 'alignment_id', 'model', 'id']
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, name, extra_keys)
    )

    # Fill lazy loading stuff
    DatasetCatalog.get(name)

    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type='coco',
        rendering_root=rendering_root,
        full_annot=full_annot,
        **metadata
    )

    # Register CAD models and categories
    register_cads(name, cad_file, scene_file, point_file, grid_file)
    register_categories(name, category_file, class_freq_method)
