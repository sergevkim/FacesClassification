from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


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
        default="{}/runs".format(Path.cwd()),
        type=str,
        help="logs dir, default: runs".format(Path.cwd()))

    return parser.parse_args()


def get_data_loaders(imgs_dir, labels_filename):
    img_filenames = [str(p) for p in Path(imgs_dir.glob('*.png'))]
    #labels = [str(p) for p in Path(imgs_dir.glob('*.png'))]
    #TODO labels file handler

    train_loader = DataLoader(
        SimpleDataset(img_filenames),
        batch_size=batch_size
    )

    loaders = {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
    }

    return loaders


class SimpleDataset:
    def __init__(self, img_filenames, img_labels):
        self.img_filenames = img_filenames
        self.labels = labels

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ToTensor()(img)
        label = self.labels[img_filename]

        return (img, label)

