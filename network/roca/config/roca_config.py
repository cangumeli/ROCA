from collections import Counter
from math import log

from roca.config import maskrcnn_config
from roca.data.constants import IMAGE_SIZE
from roca.modeling import ROCA, ROCAROIHeads


def roca_config(
    train_data: str,
    test_data: str,
    batch_size: int = 2,
    num_proposals: int = 128,
    num_classes: int = 17,
    max_iter: int = 100000,
    lr: float = 2e-2,
    workers: int = 0,
    eval_period: int = 5000,
    eval_step: bool = False,
    output_dir: str = '',  # default
    base_config: str = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
    anchor_clusters: dict = None,
    min_anchor_size: int = 64,
    noc_scale=10000,
    noc_offset=1,
    depth_scale=1000,
    class_freqs: Counter = None,
    steps: list = [60000, 80000],
    random_flip: bool = False,
    color_jitter: bool = False,
    pooler_size: int = 14,
    batch_average: bool = False,
    depth_grad_losses: bool = False,
    depth_res: tuple = IMAGE_SIZE,
    per_category_mask: bool = False,
    disable_retrieval: bool = False,
    min_nocs: int = 4*4,
    per_category_noc: bool = False,
    noc_embed: bool = False,
    noc_weights: bool = True,
    per_category_trans: bool = True,
    noc_weight_head: bool = False,
    noc_weight_skip: bool = False,
    noc_rot_init: bool = False,
    seed: int = -1,
    gclip: float = 10.0,
    augment: bool = False,
    zero_center: bool = False,
    irls_iters: int = 1,
    wild_retrieval: bool = True,
    # retrieval_mode: str = 'nearest'
    retrieval_mode: str = 'resnet_resnet+image+comp',
    confidence_thresh_test: float = 0.5,
    e2e: bool = True
):
    cfg = maskrcnn_config(
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        num_proposals=num_proposals,
        num_classes=num_classes,
        max_iter=max_iter,
        lr=lr,
        num_workers=0,
        eval_period=eval_period,
        output_dir=output_dir,
        base_config=base_config,
        custom_mask=True,
        disable_flip=True,
        enable_crop=False,
        anchor_clusters=anchor_clusters,
        min_anchor_size=min_anchor_size
    )

    # Disable resizing of any kind
    cfg.INPUT.MIN_SIZE_TRAIN = min(IMAGE_SIZE)
    cfg.INPUT.MIN_SIZE_TEST = min(IMAGE_SIZE)
    cfg.INPUT.MAX_SIZE_TRAIN = max(IMAGE_SIZE)
    cfg.INPUT.MAX_SIZE_TEST = max(IMAGE_SIZE)

    # Store NOC decoding data
    cfg.INPUT.NOC_SCALE = noc_scale
    cfg.INPUT.NOC_OFFSET = noc_offset
    cfg.INPUT.DEPTH_SCALE = depth_scale
    cfg.INPUT.DEPTH_RES = depth_res
    cfg.INPUT.AUGMENT = augment
    cfg.INPUT.CAD_TYPE = _get_cad_type(retrieval_mode)

    # Set roi heads
    cfg.MODEL.META_ARCHITECTURE = ROCA.__name__
    cfg.MODEL.ROI_HEADS.NAME = ROCAROIHeads.__name__

    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = pooler_size
    cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False
    # NOTE: below is different from non-class-agnostic mask
    # as all non-cad/non-benchmark classes share a single output
    cfg.MODEL.ROI_HEADS.PER_CATEGORY_MASK = per_category_mask

    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.MODEL.ROI_HEADS.NOC_MIN = min_nocs
    cfg.MODEL.ROI_HEADS.PER_CATEGORY_NOC = per_category_noc
    cfg.MODEL.ROI_HEADS.NOC_EMBED = noc_embed
    cfg.MODEL.ROI_HEADS.NOC_WEIGHTS = noc_weights
    cfg.MODEL.ROI_HEADS.PER_CATEGORY_TRANS = per_category_trans
    cfg.MODEL.ROI_HEADS.NOC_WEIGHT_HEAD = noc_weight_head
    cfg.MODEL.ROI_HEADS.NOC_WEIGHT_SKIP = noc_weight_skip
    cfg.MODEL.ROI_HEADS.NOC_ROT_INIT = noc_rot_init
    cfg.MODEL.ROI_HEADS.ZERO_CENTER = zero_center
    cfg.MODEL.ROI_HEADS.IRLS_ITERS = irls_iters
    cfg.MODEL.ROI_HEADS.E2E = e2e

    cfg.MODEL.ROI_HEADS.CONFIDENCE_THRESH_TEST = confidence_thresh_test


    # Set depth config
    cfg.MODEL.DEPTH_BATCH_AVERAGE = batch_average
    cfg.MODEL.DEPTH_GRAD_LOSSES = depth_grad_losses

    # Set retrieval config
    cfg.MODEL.RETRIEVAL_ON = not disable_retrieval
    cfg.MODEL.WILD_RETRIEVAL_ON = wild_retrieval
    cfg.MODEL.RETRIEVAL_MODE = retrieval_mode
    cfg.MODEL.RETRIEVAL_BASELINE = _is_baseline(retrieval_mode)

    # Set optimizer configuration
    cfg.SOLVER.STEPS = tuple(steps)
    cfg.SOLVER.WORKERS = workers
    cfg.SOLVER.CHECKPOINT_PERIOD = eval_period
    cfg.SOLVER.EVAL_STEP = eval_step
    
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = gclip > 0
    # cfg.SOLVER.CLI
    cfg.SOLVER.CLIP_VALUE = gclip

    # Set class scales
    if not class_freqs:
        class_scales = []
    else:
        class_scales = sorted((k, 1 / log(v)) for k, v in class_freqs.items())
        # class_scales = sorted((k, 1 / v) for k, v in class_freqs.items())
        ratio = 1 / max(v for k, v in class_scales)
        class_scales = [(k, v * ratio) for k, v in class_scales]
    cfg.MODEL.CLASS_SCALES = class_scales

    # Custom logic for augmentations
    cfg.INPUT.CUSTOM_FLIP = random_flip
    cfg.INPUT.CUSTOM_JITTER = color_jitter

    # Set the seed
    cfg.SEED = seed

    return cfg


def _get_cad_type(retrieval_mode: str) -> str:
    # TODO: generalize for pairs
    if 'resnet' in retrieval_mode:
        return 'voxel'
    else:
        return 'point'


def _is_baseline(retrieval_mode: str) -> bool:
    return retrieval_mode in ('random', 'nearest', 'first')
