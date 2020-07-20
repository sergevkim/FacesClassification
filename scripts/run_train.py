#import sys
#sys.path.append('..')
import os
print(os.getcwd())

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torchvision.models import resnet18

from lib.models import SimpleClassifier, ResNet
from lib.trainer import Trainer
from lib.utils import train_parse_args
from lib.utils.data import get_data_loaders
from lib.constants import HYPERPARAMETERS, PATHS


def main():
    params = vars(train_parse_args(
        hyperparameters_default=HYPERPARAMETERS,
        paths_default=PATHS))

    if not params['disable_cuda'] and torch.cuda.is_available():
        params['device'] = torch.device('cuda:0')
    else:
        params['device'] = torch.device('cpu')

    print(torch.cuda.is_available(), params['device'])

    loaders = get_data_loaders(
        imgs_dir=params['imgs_dir'],
        labels_filename=params['labels_filename'],
        batch_size=params['batch_size'],
        n_imgs=params['n_imgs'])

    model = ResNet()
    #model = SimpleClassifier()
    model.cuda()
    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = BCELoss()
    trainer = Trainer(params, model, optimizer, criterion)

    trainer.run(loaders)


if __name__ == "__main__":
    main()

