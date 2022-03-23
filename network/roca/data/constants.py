import numpy as np


BENCHMARK_CLASSES = (
    'bathtub',
    'bin',
    'bookcase',
    'chair',
    'cabinet',
    'display',
    'sofa',
    'table',
)
ALL_CLASSES = (
    'bathtub',
    'bed',
    'bin',
    'bookcase',
    'chair',
    'cabinet',
    'display',
    'sofa',
    'table',
)

SYMMETRY_CLASS_IDS = {
    '__SYM_NONE': 0,
    '__SYM_ROTATE_UP_2': 1,
    '__SYM_ROTATE_UP_4': 2,
    '__SYM_ROTATE_UP_INF': 3
}
SYMMETRY_ID_CLASSES = {v: k for k, v in SYMMETRY_CLASS_IDS.items()}

CAD_TAXONOMY = {
    2747177: 'bin',
    2808440: 'bathtub',
    2818832: 'bed',
    2871439: 'bookcase',
    2933112: 'cabinet',
    3001627: 'chair',
    3211117: 'display',
    4256520: 'sofa',
    4379243: 'table'
}
CAD_TAXONOMY_REVERSE = {v: k for k, v in CAD_TAXONOMY.items()}

# IMAGE_SIZE = (480, 640)
IMAGE_SIZE = (360, 480)

VOXEL_RES = (32, 32, 32)

COLOR_BY_CLASS = {
    2747177: np.array([210, 43, 16]) / 255,
    2808440: np.array([176, 71, 241]) / 255,
    2818832: np.array([204, 204, 255]) / 255,
    2871439: np.array([255, 191, 0]) / 255,
    2933112: np.array([255, 127, 80]) / 255,
    3001627: np.array([44, 131, 242]) / 255,
    3211117: np.array([212, 172, 23]) / 255,
    4256520: np.array([237, 129, 241]) / 255,
    4379243: np.array([32, 195, 182]) / 255
}
