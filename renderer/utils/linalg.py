import numpy as np
import quaternion
import torch
from pytorch3d.structures import Meshes


# From https://github.com/skanti/Scan2CAD/
# noinspection PyPep8Naming
def make_M_from_tqs(t: list, q: list, s: list) -> np.ndarray:
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    M = T.dot(R).dot(S)
    return M


def make_M_from_tr(t: np.ndarray, r: np.ndarray):
    mat = np.eye(4, 4)
    mat[:3, :3] = r
    mat[:3, 3] = t
    return mat


# From https://github.com/skanti/Scan2CAD/
# noinspection PyPep8Naming
def decompose_mat4(M: np.ndarray) -> tuple:
    R = M[0:3, 0:3]
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])
    s = np.array([sx, sy, sz])

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz
    q = quaternion.from_rotation_matrix(R[0:3, 0:3])
    q = quaternion.as_float_array(q)

    t = M[0:3, 3]

    return t, q, s


def to_hom(mat: np.ndarray) -> np.ndarray:
    is_1d = mat.ndim == 1
    if is_1d:
        mat = mat.reshape(1, -1)
    hom = np.concatenate([mat, np.ones((*mat.shape[:-1], 1))], axis=-1)
    return hom.reshape(-1) if is_1d else hom


def from_hom(mat: np.ndarray) -> np.ndarray:
    is_1d = mat.ndim == 1
    if is_1d:
        mat.reshape(1, -1)
    de_hom, ones = np.split(mat, [3], axis=-1)
    return de_hom.reshape(-1) if is_1d else de_hom


def transform_mesh(mesh: Meshes, transform: np.ndarray) -> Meshes:
    point_list = mesh.verts_list()
    assert len(point_list) == 1
    points = from_hom(to_hom(point_list[0].numpy()) @ transform.T)
    mesh = Meshes(
        verts=[torch.from_numpy(points).float()],
        faces=mesh.faces_list()
    )
    return mesh


def back_project(
    intr: np.ndarray,
    depth: np.ndarray,
    mask = None
) -> np.ndarray:
    if mask is not None:
        height, width = mask.shape[:2]
    else:
        height, width = depth.shape[:2]
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    x, y = np.meshgrid(x, y)
    if mask is not None:
        x, y = x[mask], y[mask]
    pts = np.stack([x * depth, y * depth, depth], axis=-1)
    return pts @ np.linalg.inv(intr[:3, :3]).T


def perspective(intr: np.ndarray, point):
    # TODO: maybe generalize to batches
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    if not isinstance(intr, np.ndarray):
        intr = np.array(intr)

    proj = from_hom(intr @ point)
    z = proj[2]
    x = proj[0] / z
    y = proj[1] / z

    return x.item(), y.item(), z.item()
