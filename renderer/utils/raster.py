import cv2 as cv
import numpy as np
from collections import namedtuple

from pytorch3d.structures import Meshes

from scan2cad_rasterizer import Rasterizer as _Rasterizer
from utils.linalg import back_project

try:
    import open3d as o3d
except ImportError:
    o3d = None


COLOR_BY_CLASS = {
    # 2747177: np.array([241, 43, 16]) / 255,
    2747177: np.array([241, 43, 16]) / 255,
    2808440: np.array([176, 71, 241]) / 255,
    2818832: np.array([204, 204, 255]) / 255,
    2871439: np.array([255, 191, 0]) / 255,
    2933112: np.array([255, 127, 80]) / 255,
    3001627: np.array([44, 131, 242]) / 255,
    3211117: np.array([212, 172, 23]) / 255,
    4256520: np.array([237, 129, 241]) / 255,
    4379243: np.array([32, 195, 182]) / 255,
}
COLOR_BY_CLASS_BRG = {k: v[::-1] for k, v in COLOR_BY_CLASS.items()}


def to_o3d(meshes: Meshes, compute_normals=True, color=None):
    assert len(meshes) == 1
    assert o3d is not None, 'No open3d!'
    faces = meshes.faces_list()[0].numpy()
    verts = meshes.verts_list()[0].numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    if compute_normals:
        mesh.compute_vertex_normals()
    if color is not None:
        mesh.paint_uniform_color(color)
    return mesh


class Rasterizer(_Rasterizer):
    def __init__(
        self,
        width: int,
        height: int,
        intrinsics,
        rendering_nocs: bool = False,
        has_normals: bool = False
    ):
        if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (4, 4):
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        elif isinstance(intrinsics, tuple) and len(intrinsics) == 4:
            fx, fy, cx, cy = intrinsics
        else:
            raise RuntimeError(
                'intrinsics must be a 4x4 numpy array ' +
                'or a tuple (fx, fy, cx, cy)'
            )

        self.used_idx = set()
        super().__init__(
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            rendering_nocs,
            has_normals
        )

    def clear_models(self):
        self.used_idx.clear()
        super().clear_models()

    def add_mesh(self, mesh: Meshes, idx: int, to_noc: np.ndarray, normals=None):
        if self.has_normals and normals is None:
            normals = mesh.faces_normals_list()[0].numpy()
            # import pdb; pdb.set_trace()
        else:
            normals = np.array([])
        self.add_model(
            mesh.faces_list()[0].numpy(),
            mesh.verts_list()[0].numpy(),
            idx,
            to_noc,
            normals
        )

    def add_model(
        self,
        triangles: np.ndarray,
        vertices: np.ndarray,
        idx: int,
        to_noc: np.ndarray,
        normals: np.ndarray
    ):
        assert idx not in self.used_idx, '{} was already used'.format(idx)

        if triangles.dtype != self.index_dtype:
            triangles = triangles.astype(self.index_dtype)

        if vertices.dtype != self.scalar_dtype:
            vertices = vertices.astype(self.scalar_dtype)

        idx = getattr(np, self.idx_dtype)(idx)

        if to_noc.dtype != self.scalar_dtype:
            to_noc = to_noc.astype(self.scalar_dtype)

        if normals.dtype != self.scalar_dtype:
            normals = normals.astype(self.scalar_dtype)

        if self.has_normals:
            assert normals.shape == triangles.shape

        super().add_model(triangles, vertices, idx, to_noc, normals)

        self.used_idx.add(idx)



class NormalShader:
    def __init__(self, normal_coef=0.3, blur=(1, 1)):
        self.normal_coef = normal_coef
        self.blur = blur

    def __call__(
        self,
        normals: np.ndarray,
        instances: np.ndarray,
        colors: dict
    ) -> tuple:

        image = np.zeros_like(normals)
        mask = instances > 0
        normals, instances = normals[mask], instances[mask]
        
        color = np.zeros_like(normals)
        for i in np.unique(instances):
            color[instances == i] = colors[i]

        image[mask] = (
            (1 - self.normal_coef) * color
            + self.normal_coef * (normals * color).sum(-1, keepdims=True)
        )

        if self.blur != (1, 1):
            image = cv.GaussianBlur(
                image, self.blur, 1 / (2 * np.pi * self.blur[0]**2)
            )

        return image, mask
