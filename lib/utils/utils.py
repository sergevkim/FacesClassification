from argparse import ArgumentParser
from pathlib import Path


def train_parse_args(params):
    parser = ArgumentParser()
    parser.add_argument(
        '--batch-size',
        default=params['batch_size'],
        type=int,
        help=f"batch_size, default: {params['batch_size']}")
    parser.add_argument(
        '--checkpoints-dir',
        default=params['checkpoints_dir'],
        type=str,
        help=f"checkpoints dir, default: {params['checkpoints_dir']}")
    parser.add_argument(
        '--checkpoint-filename',
        default=params['checkpoint_filename'],
        type=str,
        help=f"filename, default: {params['labels_filename']}")
    parser.add_argument(
        '--disable-cuda',
        action='store_true',
        help="disable_cuda flag, by defaut it fits on cuda")
    parser.add_argument(
        '--imgs-dir',
        default=params['imgs_dir'],
        type=str,
        help=f"imgs dir, default: {params[imgd_dir]}")
    parser.add_argument(
        '--label',
        default=params['label'],
        type=str,
        help=f"imgs dir, default: {params[label]}")
    parser.add_argument(
        '--labels-filename',
        default=params['labels_filename'],
        type=str,
        help=f"labels description, default: {params['labels_filename']}")
    parser.add_argument(
        '--logs-dir',
        default=params['logs_dir'],
        type=str,
        help=f"logs dir, default: {params['logs_dir']}")
    parser.add_argument(
        '--n-epochs',
        default=params['n_epochs'],
        type=int,
        help=f"n_epochs, default: {params['n_epochs']}")
    parser.add_argument(
        '--n-imgs',
        default=params['n_imgs'],
        type=int,
        help=f"n_imgs, default: {params['n_imgs']}")
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="verbose flag")
    parser.add_argument(
        '--version',
        default=params['version'],
        type=str,
        help=f"version, default: {params['version']}")

    return parser.parse_args()

