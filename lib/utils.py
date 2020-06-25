from argparse import ArgumentParser
from pathlib import Path

import torch


def train_parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--batch-size',
        default=64,
        type=int,
        help="batch_size, default: 64")
    parser.add_argument(
        '--checkpoints-dir',
        default="{}/runs".format(Path.cwd()),
        type=str,
        help="checkpoints dir, default: checkpoints".format(Path.cwd()))
    parser.add_argument(
        '--logs-dir',
        default="{}/logs".format(Path.cwd()),
        type=str,
        help="logs dir, default: logs".format(Path.cwd()))

    return vars(parser.parse_args())


def get_data_loaders():
    pass

