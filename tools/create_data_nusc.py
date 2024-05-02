# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from data_converter import nuscenes_converter as nuscenes_converter



def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       max_sweeps=10,
                       with_circular_cams=False):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    if with_circular_cams:
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT',
        ]
        print("using circular cam order:")
        print(camera_types)
        info_prefix = f"{info_prefix}_with_circular_cams"
    else:
        camera_types=None
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps,
        cam_order=camera_types)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--trainval-path',
    type=str,
    default='./data/nuscenes',
    help='specify the trainval path of dataset')
parser.add_argument(
    '--test-path',
    type=str,
    default='./data/nuscenes',
    help='specify the test path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='/data/nuscenes',
    required=False,
    help='name of info pkl')
parser.add_argument(
    '--with_circular_cams',
    action='store_true',
    help='whether to use circular cam order'
)
parser.add_argument('--extra-tag', type=str, default='nuscenes2d')
args = parser.parse_args()

if __name__ == '__main__':
    if args.version != 'v1.0-mini':
        print(f"loading trainval from: {args.trainval_path}\nloading test from: {args.test_path}")
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.trainval_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps,
            with_circular_cams=args.with_circular_cams)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.test_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps,
            with_circular_cams=args.with_circular_cams)
    elif args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps,
            with_circular_cams=args.with_circular_cams)
