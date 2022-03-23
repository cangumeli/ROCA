import argparse
import cv2 as cv
import multiprocessing as mp
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from shutil import copyfile


def resize_image(
    image: np.ndarray,
    short_scale: int,
    long_scale: int,
    interpolation,
    intrinsics: np.ndarray = None,
    blur: bool = False
):
    update_intr = intrinsics is not None
    if image.shape[:2] == (short_scale, long_scale):
        return image if not update_intr else (image, intrinsics)
    unsqueeze = image.ndim == 2

    if update_intr:
        intrinsics = intrinsics.copy()
        intrinsics[:2, :3] *= short_scale / image.shape[0]

    h = short_scale
    w = round(short_scale * image.shape[1] / image.shape[0])

    if blur:
        rath = 2 * round(image.shape[0] / h) + 1
        ratw = 2 * round(image.shape[1] / w) + 1
        image = cv.GaussianBlur(
            image,
            (ratw, rath),
            sigmaX=(ratw-1)/6,
            sigmaY=(rath-1)/6
        )

    image = cv.resize(image, (w, h), interpolation=interpolation)
    image = cv.medianBlur(image, 3)

    # Cropping!
    if unsqueeze:
        image = image.reshape(*image.shape, 1)

    if w != long_scale:
        assert long_scale < w, 'Padding is not supported yet!'
        crop = (w - long_scale) // 2
        image = image[:, crop:crop+long_scale, :]
        if update_intr:
            intrinsics[0, 2] -= crop

    if unsqueeze:
        image = image.squeeze()

    return (image, intrinsics) if update_intr else image


def worker(args, data_queue):
    while True:
        try:
            scene = data_queue.get(timeout=20)
        except Empty:
            break 

        print('Process {} is processing {}'.format(os.getpid(), scene), flush=True)

        current_dir = os.path.join(args.image_root, scene)
        # print('{}: {} ({} / {})'
        #      .format(datetime.now(), scene, index + 1, len(scenes)))

        output_dir = os.path.join(args.output_root, scene)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(current_dir, 'intrinsics_color.txt')) as f:
        # with open(os.path.join(current_dir, 'intrinsic_color.txt')) as f:
            intrinsics = np.array([
                [float(w) for w in line.strip().split()]
                for line in f
            ])

        color_dir = os.path.join(current_dir, 'color')
        pose_dir = os.path.join(current_dir, 'pose')

        color_output_dir = os.path.join(output_dir, 'color')
        Path(color_output_dir).mkdir(parents=True, exist_ok=True)
        pose_output_dir = os.path.join(output_dir, 'pose')
        Path(pose_output_dir).mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(os.listdir(color_dir)):
            if '.jpg' not in img:
                continue

            pose_file = img.replace('jpg', 'txt')
            copyfile(
                src=os.path.join(pose_dir, pose_file),
                dst=os.path.join(pose_output_dir, pose_file)
            )

            # Copy the pose directly
            # cv.namedWindow('Image')
            # cv.namedWindow('Resized')
            image = cv.imread(os.path.join(color_dir, img))
            # cv.imshow('Image', image)
            if i == 0:
                image, intrinsics = resize_image(
                    image=image,
                    short_scale=args.short_scale,
                    long_scale=args.long_scale,
                    # interpolation=cv.INTER_CUBIC,
                    interpolation=cv.INTER_NEAREST,
                    blur=True,
                    intrinsics=intrinsics
                )
                np.around(intrinsics, decimals=2, out=intrinsics)
                intrinsics_output_file = os.path.join(
                    output_dir, 'intrinsics_color.txt'
                )
                with open(intrinsics_output_file, 'w') as f:
                    for row in intrinsics.tolist():
                        f.write(' '.join(map(str, row)) + '\n')
            else:
                image = resize_image(
                    image=image,
                    short_scale=args.short_scale,
                    long_scale=args.long_scale,
                    blur=True,
                    interpolation=cv.INTER_NEAREST  # cv.INTER_CUBIC
                )
            cv.imwrite(os.path.join(color_output_dir, img), image)

        if args.instance:
            instance_dir = os.path.join(current_dir, 'instance')
            instance_output_dir = os.path.join(output_dir, 'instance')
            Path(instance_output_dir).mkdir(parents=True, exist_ok=True)
            for img in os.listdir(instance_dir):
                if '.png' not in img:
                    continue
                instance = cv.imread(os.path.join(instance_dir, img), -1)
                instance = resize_image(
                    image=instance,
                    short_scale=args.short_scale,
                    long_scale=args.long_scale,
                    interpolation=cv.INTER_NEAREST
                )
                # instance = cv.medianBlur(instance, 5)
                cv.imwrite(os.path.join(instance_output_dir, img), instance)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--short_scale', type=int, default=360)
    parser.add_argument('--long_scale', type=int, default=480)
    parser.add_argument('--instance', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args(args)
    print(args)
    print(datetime.now())

    assert args.image_root != args.output_root, \
        'Overriding the same directory is not allowed!'
    args.image_root = os.path.join(
        args.image_root, 'tasks', 'scannet_frames_25k'
    )
    args.output_root = os.path.join(
        args.output_root, 'tasks', 'scannet_frames_25k'
    )

    scenes = sorted(os.listdir(args.image_root))
    queue = (mp.Queue if args.num_workers else Queue)(maxsize=len(scenes))
    for scene in scenes:
        if 'scene' not in scene:
            continue
        queue.put(scene)

    if args.num_workers:
        workers = [
            mp.Process(target=worker, args=(args, queue))
            for _ in range(args.num_workers)
        ]
        for process in workers:
            process.start()
        for process in workers:
            process.join()
    else:
        worker(args, queue)

    print('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])
