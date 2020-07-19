from pathlib import Path
from PIL import Image

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize

from lib.constants import SELECTED_FEATURES

import matplotlib.pyplot as plt #TODO remove


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


def prepare_labels(labels_filename, n_imgs, label_number):
    labels_file = open(labels_filename, 'r')
    #n_filenames = int(labels_file.readline())
    #all_features = labels_file.readline().split()
    labels = dict()

    for i in range(n_imgs):
        string = labels_file.readline().split()
        img_number = int(string[0].split('.')[0])
        img_number_six_symbols = '0' * (6 - len(str(img_number))) + str(img_number)  # example: 142 -> 000142
        img_filename = f"/home/sergevkim/git/FacesClassification/data/img_align_celeba/{img_number_six_symbols}.jpg" #TODO imgs_dir
        if 1 <= img_number <= n_imgs:
            labels[img_filename] = float(string[label_number])

    labels_file.close()

    return labels


def get_data_loaders(imgs_dir, labels_filename, batch_size, n_imgs, label):
    img_filenames = [str(p) for p in Path(imgs_dir).glob('*.jpg')]
    img_filenames.sort(key=lambda x : int(x.split('/')[-1][:-4]))
    img_filenames = img_filenames[:n_imgs]
    
    label_number = SELECTED_FEATURES[label]
    labels = prepare_labels(
        labels_filename=labels_filename,
        n_imgs=n_imgs,
        label_number=label_number)
    
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
        img = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])(img)
        label = self.img_labels[img_filename]
        
        #if idx in [0, 1, 2, 3, 4, 5, 1000, 1001]:
        #    print(idx, img_filename, label)

        return (img, label)
