from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from lib.constants import SELECTED_FEATURES


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


def log_grad_norm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())
    print('!')


def prepare_labels(labels_filename, img_filenames, n_imgs, label_number):
    labels_file = open(labels_filename, 'r')
    n_filenames = int(labels_file.readline())
    all_features = labels_file.readline().split()
    labels = dict()

    for i in range(n_filenames):
        string = labels_file.readline().split()
        img_number = int(string[0].split('.')[0])
        img_filename = f"/home/sergevkim/git/FacesClassification/data/CelebaHQ/{img_number}.png"
        if 1 <= img_number <= 30000:
            labels[img_filename] = float(string[label_number]) # 5 index is bald feature

    labels_file.close()

    return labels


def get_data_loaders(imgs_dir, labels_filename, batch_size, n_imgs, label='Male'):
    img_filenames = [str(p) for p in Path(imgs_dir).glob('*.png')]
    label_number = SELECTED_FEATURES[label]
    labels = prepare_labels(labels_filename, img_filenames, n_imgs, label_number)

    train_img_filenames, valid_img_filenames = train_test_split(
        img_filenames,
        test_size=0.1)

    train_dataset = SimpleDataset(
        img_filenames=train_img_filenames,
        img_labels=labels)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size)

    valid_dataset = SimpleDataset(
        img_filenames=valid_img_filenames,
        img_labels=labels)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size)

    loaders = {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
    }

    return loaders


class SimpleDataset:
    def __init__(self, img_filenames, img_labels):
        self.img_filenames = img_filenames
        self.img_labels = img_labels

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img = Image.open(img_filename)
        img = ToTensor()(img)
        label = self.img_labels[img_filename]

        return (img, label)
