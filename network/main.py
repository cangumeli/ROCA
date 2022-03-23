import argparse
import json
import os.path as path
import sys
from os import makedirs

from roca.config import roca_config
from roca.data import CategoryCatalog
from roca.data.datasets import register_scan2cad
from roca.engine import Trainer


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--rendering_root', required=True)
    parser.add_argument('--full_annot', required=True)

    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--override_output', default=0, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_iter', default=80000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_proposals', default=128, type=int)
    parser.add_argument('--eval_period', default=2500, type=int)
    parser.add_argument(
        '--freq_scale', choices=['none', 'image', 'cad'], default='image'
    )
    parser.add_argument('--steps', nargs='+',
                        default=[60000], type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--eval_step', default=0, type=int)
    parser.add_argument('--gclip', type=float, default=10.0)
    parser.add_argument('--augment', default=1, type=int)

    parser.add_argument('--pooler_size', default=16, type=int)
    parser.add_argument('--batch_average', default=0, type=int)
    parser.add_argument('--depth_grad_losses', default=0, type=int)
    parser.add_argument('--per_category_mask', default=1, type=int)
    parser.add_argument('--enable_nocs', default=1, type=int)
    parser.add_argument('--per_category_noc', default=0, type=int)
    parser.add_argument('--noc_embed', default=0, type=int)
    parser.add_argument('--noc_weights', default=1, type=int)
    parser.add_argument('--per_category_trans', default=1, type=int)
    parser.add_argument('--custom_noc_weights', default=1, type=int)
    parser.add_argument('--noc_weight_skip', default=0, type=int)
    parser.add_argument('--noc_rot_init', default=0, type=int)
    parser.add_argument('--zero_center', default=0, type=int)
    parser.add_argument('--irls_iters', default=1, type=int)
    parser.add_argument('--retrieval_mode', default='resnet_resnet+image+comp')
    parser.add_argument('--wild_retrieval', type=int, default=0)
    parser.add_argument('--confidence_thresh_test', type=float, default=0.5)
    parser.add_argument('--e2e', type=int, default=1)

    parser.add_argument('--eval_only', default=0, type=int)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=0, type=int)

    parser.add_argument('--seed', default=2021, type=int)

    args = parser.parse_args(sys.argv[1:] if args is None else args)
    print(args)
    return args


def register_data(args):
    data_dir = args.data_dir
    train_name = 'Scan2CAD_train'
    val_name = 'Scan2CAD_val'

    register_scan2cad(
        name=train_name,
        split='train',
        data_dir=data_dir,
        metadata={'scenes': path.abspath('../metadata/scannetv2_train.txt')},
        image_root=args.image_root,
        rendering_root=args.rendering_root,
        full_annot=args.full_annot,
        class_freq_method=args.freq_scale
    )
    register_scan2cad(
        name=val_name,
        split='val',
        data_dir=data_dir,
        metadata={'scenes': path.abspath('../metadata/scannetv2_val.txt')},
        image_root=args.image_root,
        rendering_root=args.rendering_root,
        full_annot=args.full_annot
    )

    return train_name, val_name


def make_config(train_name, val_name, args):
    cfg = roca_config(
        train_data=train_name,
        test_data=val_name,
        batch_size=args.batch_size,
        num_proposals=args.num_proposals,
        num_classes=CategoryCatalog.get(train_name).num_classes,
        max_iter=args.max_iter,
        lr=args.lr,
        output_dir=args.output_dir,
        eval_period=args.eval_period,
        eval_step=bool(args.eval_step),
        workers=args.workers,
        class_freqs=CategoryCatalog.get(train_name).freqs,
        steps=args.steps,
        pooler_size=args.pooler_size,
        batch_average=bool(args.batch_average),
        depth_grad_losses=bool(args.depth_grad_losses),
        per_category_mask=bool(args.per_category_mask),
        per_category_noc=bool(args.per_category_noc),
        noc_embed=bool(args.noc_embed),
        noc_weights=bool(args.noc_weights),
        per_category_trans=bool(args.per_category_trans),
        noc_weight_head=bool(args.custom_noc_weights),
        noc_weight_skip=bool(args.noc_weight_skip),
        noc_rot_init=bool(args.noc_rot_init),
        seed=args.seed,
        gclip=args.gclip,
        augment=bool(args.augment),
        zero_center=bool(args.zero_center),
        irls_iters=args.irls_iters,
        retrieval_mode=args.retrieval_mode,
        wild_retrieval=bool(args.wild_retrieval),
        confidence_thresh_test=args.confidence_thresh_test,
        e2e=bool(args.e2e)
    )

    # NOTE: Training state will be reset in this case!
    if args.checkpoint.lower() not in ('', 'none'):
        cfg.MODEL.WEIGHTS = args.checkpoint

    return cfg


def setup_output_dir(args, cfg):
    output_dir = args.output_dir
    assert not args.resume or path.isdir(args.output_dir), \
        'No backup found in {}'.format(args.output_dir)
    makedirs(output_dir, exist_ok=args.override_output or args.resume)

    if not args.eval_only and not args.resume:
        # Save command line arguments
        with open(path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)
        # Save the config as yaml
        with open(path.join(output_dir, 'config.yaml'), 'w') as f:
            cfg.dump(stream=f)


def train_or_eval(args, cfg):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if args.eval_only:
        trainer.test(cfg, trainer.model)
    elif args.resume:
        trainer.test(cfg, trainer.model)
        trainer.train()
    else:
        trainer.train()


def main(args):
    train_name, val_name = register_data(args)
    cfg = make_config(train_name, val_name, args)
    setup_output_dir(args, cfg)
    train_or_eval(args, cfg)


if __name__ == '__main__':
    main(parse_args())
