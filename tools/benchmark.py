# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
import os
import sys
sys.path.append('./')
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--samples', type=int, default=300, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    ## add samples_per_gpu=batch_size
    cfg.data.setdefault("samples_per_gpu", cfg['batch_size'])
    ## add debug args
    if 'debug_modules' in cfg:
        cfg['debug_args']['debug_modules'] = cfg.debug_modules
        cfg.model.setdefault("debug_args", cfg['debug_args'])

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # load_checkpoint(model, args.checkpoint, map_location='cpu')


    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = args.warmup
    pure_inf_time = 0

    memory_allocated_all = []
    memory_reserved_all = []
    max_memory_reserved_all=[]

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time


        if i >= num_warmup:
            pure_inf_time += elapsed
            after_memory_allocated=torch.cuda.memory_allocated(0)
            after_memory_reserved=torch.cuda.memory_reserved(0)
            after_max_memory_reserved=torch.cuda.max_memory_reserved(0)
            memory_allocated_all.append(after_memory_allocated)
            memory_reserved_all.append(after_memory_reserved)
            max_memory_reserved_all.append(after_max_memory_reserved)
            if (i + 1) % args.log_interval == 0:
                avg_mem_allocated = sum(memory_allocated_all) / (i+1-num_warmup)
                avg_mem_reserved = sum(memory_reserved_all) / (i+1-num_warmup)
                avg_max_mem_reserved = sum(max_memory_reserved_all) / (i+1-num_warmup)
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s, '
                      f'(mem allocated: {avg_mem_allocated:.6f}, '
                      f'mem reserved: {avg_mem_reserved:.6f}, '
                      f'max mem reserved: {avg_max_mem_reserved:.6f})'
                      )

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break


if __name__ == '__main__':
    main()
