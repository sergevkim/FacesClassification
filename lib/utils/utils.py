from argparse import ArgumentParser
from pathlib import Path


def train_parse_args(hyperparameters_default, paths_default):
    parser = ArgumentParser()

    parser.add_argument(
        '--batch-size',
        default=hyperparameters_default['batch_size'],
        type=int,
        help=f"batch_size, default: {hyperparameters_default['batch_size']}")
    parser.add_argument(
        '--disable-cuda',
        action='store_true',
        help="disable_cuda flag, by defaut it fits on cuda")
    parser.add_argument(
        '--label',
        default=hyperparameters_default['label'],
        type=str,
        help=f"label, default: {hyperparameters_default['label']}") #TODO train on all labels?
    parser.add_argument(
        '--n-epochs',
        default=hyperparameters_default['n_epochs'],
        type=int,
        help=f"n_epochs, default: {hyperparameters_default['n_epochs']}")
    parser.add_argument(
        '--n-imgs',
        default=hyperparameters_default['n_imgs'],
        type=int,
        help=f"n_imgs < 30000, default: {hyperparameters_default['n_imgs']}")
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="verbose flag")
    parser.add_argument(
        '--version',
        default=hyperparameters_default['version'],
        type=str,
        help=f"version, default: {hyperparameters_default['version']}")

    parser.add_argument(
        '--checkpoints-dir',
        default=paths_default['checkpoints_dir'],
        type=str,
        help=f"checkpoints dir, default: {paths_default['checkpoints_dir']}")
    parser.add_argument(
        '--checkpoint-filename',
        default=paths_default['checkpoint_filename'],
        type=str,
        help=f"start train from <checkpoint filename>, default: {paths_default['checkpoint_filename']} (empty)")
    parser.add_argument(
        '--imgs-dir',
        default=paths_default['imgs_dir'],
        type=str,
        help=f"imgs dir, default: {paths_default['imgs_dir']}")
    parser.add_argument(
        '--labels-filename',
        default=paths_default['labels_filename'],
        type=str,
        help=f"labels description, default: {paths_default['labels_filename']}")
    parser.add_argument(
        '--logs-dir',
        default=paths_default['logs_dir'],
        type=str,
        help=f"logs dir, default: {paths_default['logs_dir']}")

    return parser.parse_args()

