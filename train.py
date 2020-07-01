import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torchvision.models import resnet18

from lib.models import SimpleClassifier, ResNet, get_resnet
from lib.trainer import Trainer
from lib.utils import train_parse_args, get_data_loaders
from lib.constants import get_hyperparameters


def main():
    params_default = get_hyperparameters()
    params = vars(train_parse_args(params_default))
    print(params)
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

