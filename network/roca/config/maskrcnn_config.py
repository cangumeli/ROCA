from detectron2 import model_zoo
from detectron2.config import get_cfg


def _set_anchors(cfg, anchor_clusters: dict, min_anchor_size: int):
    if anchor_clusters is None:
        return

    anchor_sizes = [[el] for el in anchor_clusters['size_centers']]
    # TODO: make the replacement below optional
    anchor_sizes[-1] = [anchor_clusters['size_bounds'][-1]]
    if min_anchor_size > 0:
        anchor_sizes = [[min_anchor_size]] + anchor_sizes
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = anchor_sizes

    aspect_ratios = [anchor_clusters['ratio_centers']]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = aspect_ratios


def maskrcnn_config(
    train_data: str,
    test_data: str,
    batch_size: int = 2,
    num_proposals: int = 128,
    num_classes: int = 17,
    max_iter: int = 100000,
    lr: float = 2e-2,
    num_workers: int = 2,
    eval_period: int = 5000,
    output_dir: str = '',  # default
    base_config: str = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
    custom_mask: bool = False,
    disable_flip: bool = False,
    enable_crop: bool = False,
    anchor_clusters: dict = None,
    min_anchor_size: int = 64,
    gclip=False,
    gclip_type='norm',
    gclip_value=10.0,
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_config))

    cfg.DATASETS.TRAIN = (train_data,)
    cfg.DATASETS.TEST = (test_data,)
    cfg.DATALOADER.NUM_WORKERS = num_workers

    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = gclip
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = gclip_type
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = gclip_value

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = num_proposals
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    _set_anchors(cfg, anchor_clusters, min_anchor_size)

    cfg.TEST.EVAL_PERIOD = eval_period

    if enable_crop:
        cfg.INPUT.CROP.ENABLED = True
    if custom_mask or disable_flip:
        # TODO: ensure flip is supported with custom mask
        cfg.INPUT.RANDOM_FLIP = 'none'

    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.INPUT.CUSTOM_MASK = custom_mask

    if output_dir != '':
        cfg.OUTPUT_DIR = output_dir

    return cfg
