import numpy as np
import os
import pickle as pkl
import sys
from itertools import product
from trimesh import Trimesh
from trimesh.voxel.creation import local_voxelize
from tqdm import tqdm


if __name__ == '__main__':
    data_dir = sys.argv[1]

    for s in ('train', 'val'):
        cad_path = os.path.join(data_dir, 'scan2cad_{}_cads.pkl'.format(s))

        with open(cad_path, 'rb') as f:
            data = pkl.load(f)

        grid_size = 32
        indices = product(range(grid_size), range(grid_size), range(grid_size))
        indices = np.array(list(indices), dtype=np.int32)

        arrays = {}
        for datum in tqdm(data, dynamic_ncols=True):
            mesh = Trimesh(vertices=datum['verts'], faces=datum['faces'])
            grid = local_voxelize(
                mesh=mesh,
                point=np.zeros(3),
                pitch=(1 / grid_size),
                radius=(grid_size // 2),
                fill=False
            )
            grid = np.asarray(grid.matrix).transpose((2, 1, 0))
            grid = grid[:grid_size, :grid_size, :grid_size]

            arrays[(datum['catid_cad'], datum['id_cad'])] = \
                indices[grid.reshape(-1)]

        output_path = os.path.join(
            data_dir, '{}_grids_{}.pkl'.format(s, grid_size)
        )
        with open(output_path, 'wb') as f:
            pkl.dump(arrays, f)
