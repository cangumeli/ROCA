import cv2 as cv
import numpy as np
import os
import pathlib
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes


def load_mesh(obj_file_path: str, as_numpy_tuple: bool = False) -> Meshes:
    with open(obj_file_path, 'rb') as f:
        if as_numpy_tuple:
            mesh = load_obj(f, load_textures=False)
            mesh = (mesh[0].numpy(), mesh[1].verts_idx.numpy())
        else:
            mesh = load_objs_as_meshes([f], load_textures=False)
    return mesh


def load_poses(scene_dir: str) -> dict:
    poses = {}
    pose_dir = os.path.join(scene_dir, 'pose')
    for file_name in os.listdir(pose_dir):
        with open(os.path.join(pose_dir, file_name)) as f:
            pose = np.array([
                [float(e) for e in line.strip().split()]
                for line in f
            ])
        poses[file_name.replace('.txt', '')] = pose
    return poses


def load_intrinsics(scene_dir: str, image_type='color'):
    assert image_type in ['color', 'depth']
    file_name = 'intrinsics_{}.txt'.format(image_type)
    path = os.path.join(scene_dir, file_name)  # path for 25k

    # Support custom sampling
    if not os.path.isfile(path):
        new_path = os.path.join(scene_dir, 'intrinsic', file_name)
        assert os.path.isfile(new_path), 'Could not find {} or {}'.format(
            path, new_path
        )
        path = new_path

    with open(path) as f:
        intrinsics = np.array([
            [float(e) for e in line.strip().split()]
            for line in f
        ])
    return intrinsics


def load_image_size(scene_dir: str, image_type='color'):
    assert image_type in ['color', 'depth']
    file_type = '.png' if image_type == 'depth' else '.jpg'
    image_path = os.path.join(scene_dir, image_type)
    sample_file = next(
        file for file in os.listdir(image_path) if file_type in file
    )
    sample_file = os.path.join(image_path, sample_file)

    if image_type == 'depth':
        image = cv.imread(sample_file, -1)
    else:
        image = cv.imread(sample_file)
    return image.shape[:2]


def write_images(
    output_scene_dir: str,
    output_image_name: str,
    noc: np.ndarray,
    depth: np.ndarray,
    idx: np.ndarray,
    noc_offset: float = 1,
    noc_scale: int = 10000,
    depth_scale: int = 1000,
    idx_pixel_type: np.dtype = np.uint8,
    modify_buffers=True
):
    if not modify_buffers:
        depth, idx = depth.copy(), idx.copy()
        if noc is not None:
            noc = noc.copy()

    noc_dir = os.path.join(output_scene_dir, 'noc')
    depth_dir = os.path.join(output_scene_dir, 'depth')
    idx_dir = os.path.join(output_scene_dir, 'instance')
    for image_dir in (noc_dir, depth_dir, idx_dir):
        if image_dir == noc_dir and noc == None:
            continue
        pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)

    if noc is not None:
        noc += noc_offset
        noc *= noc_scale
        np.around(noc, out=noc)
        noc_path = os.path.join(noc_dir, output_image_name + '.png')
        cv.imwrite(noc_path, noc.astype(np.uint16))
    # import pdb; pdb.set_trace()

    depth *= depth_scale
    np.around(depth, out=depth)
    depth_path = os.path.join(depth_dir, output_image_name + '.png')
    cv.imwrite(depth_path, depth.astype(np.uint16))

    idx_path = os.path.join(idx_dir, output_image_name + '.png')
    cv.imwrite(idx_path, idx.astype(idx_pixel_type))
    # return noc_path, depth_path, idx_path
