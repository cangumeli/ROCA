import numpy as np
import quaternion
import torch


# See https://github.com/skanti/Scan2CAD/blob/master/Routines/Script/EvaluateBenchmark.py
def calc_rotation_diff(q: np.quaternion, q00: np.quaternion) -> float:
    np.seterr(all='raise')
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def rotation_diff(q: np.quaternion, q_gt: np.quaternion, sym='') -> float:
    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    else:
        error_rotation = calc_rotation_diff(q, q_gt)
    return error_rotation


def scale_ratio(pred_scale: torch.Tensor, gt_scale: torch.Tensor) -> float:
    # return torch.abs(torch.mean(pred_scale / gt_scale) - 1).item()
    return np.abs(np.mean(pred_scale.numpy() / gt_scale.numpy()) - 1).item()


def translation_diff(pred_trans: torch.Tensor, gt_trans: torch.Tensor) -> float:
    # return torch.norm(pred_trans - gt_trans)
    return np.linalg.norm(pred_trans.numpy() - gt_trans.numpy(), ord=2)
