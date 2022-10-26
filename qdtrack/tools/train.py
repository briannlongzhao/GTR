import argparse
import copy
import os
import os.path as osp
import time
import pathlib

import mmcv
import torch
from torch import nn
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset

from qdtrack import __version__
from qdtrack.utils import collect_env, get_root_logger

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    table_cat = PrettyTable(["Category", "Parameters"])
    total_params = 0
    cat_dict = {}
    for name, parameter in model.named_parameters():
        category = name.split('.')[-1]
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        if category in cat_dict.keys():
            cat_dict[category] += params
        else:
            cat_dict[category] = params
        total_params += params
    print(table)
    for k,v in cat_dict.items():
        table_cat.add_row([k,v])
    print(table_cat)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def compute_neuron_count(model):
    neuron = []
    h_in = 224
    w_in = 224
    named_parameters = dict(model.named_parameters())
    print(named_parameters.keys())
    for name, l in model.named_modules():
        try:
            parameter = named_parameters[name+".weight"]
        except:
            print(name)
            continue
        if not parameter.requires_grad: continue
        if isinstance(l, nn.Conv2d):
            c_in = l.in_channels
            k = l.kernel_size[0]
            h_out = int((h_in-k+2*l.padding[0])/(l.stride[0])) + 1
            w_out = int((w_in-k+2*l.padding[0])/(l.stride[0])) + 1
            c_out = l.out_channels
            neuron_count = h_out*w_out*c_out
            neuron.append(neuron_count)
            h_in = h_out
            w_in = w_out
            print('{}, neuron count:{}'.format(name, neuron_count))
        elif isinstance(l, nn.Linear):
            neuron_count = l.out_features
            neuron.append(neuron_count)
            print('{}, neuron count:{}'.format(name, neuron_count))
        elif isinstance(l, nn.AvgPool2d):
            h_in = h_in//l.kernel_size
            w_in = w_in//l.kernel_size
        elif isinstance(l, nn.modules.batchnorm.BatchNorm2d):
            #neuron_count = l.num_features
            neuron.append(neuron_count)
            print('{}, neuron count:{}'.format(name, neuron_count))
        elif isinstance(l, nn.modules.normalization.GroupNorm):
            #neuron_count = l.num_channels
            neuron.append(neuron_count)
            print('{}, neuron count:{}'.format(name, neuron_count))
        else:
            print(type(l))
    print('{:e}'.format(sum(neuron)))



def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--data-root', help='dataset root to override in the cfg', default=None)
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # note: this is a hack to change the config data_root to local ssd for slurm jobs
    if args.data_root is not None and os.getenv('TMPDIR', None) is not None:
        tmp_config = f"{os.getenv('TMPDIR')}/tmp_config.py"
        tmp_lines = list()
        with open(args.config, 'r') as fh:
            for idx, line in enumerate(fh.readlines()):
                if 'data_root = ' in line:
                    line = f"data_root = '{args.data_root}'"
                tmp_lines.append(line)
        with open(tmp_config, 'w') as fh:
            for line in tmp_lines:
                fh.write(f"{line.strip()}\n")
        time.sleep(2)

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('USE_MMDET', False):
        from mmdet.apis import train_detector as train_model
        from mmdet.models import build_detector as build_model
    else:
        from qdtrack.apis import train_model
        from qdtrack.models import build_model

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    elif (pathlib.Path(cfg.work_dir)/'latest.pth').is_file():
        cfg.resume_from = f"{cfg.work_dir}/latest.pth"
        print("loading previous state from ", cfg.resume_from)
    else:
        print("not loading previous model")
            
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.get('dist_params', dict(backend='nccl')))

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()


    compute_neuron_count(model)
    count_parameters(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save qdtrack version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            qdtrack_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
