from torch.nn import BCELoss
from torch.optim import Adam

from lib.models import SimpleClassifier
from lib.trainer import Trainer
from lib.utils import train_parse_args, get_data_loaders


def main():
    params = vars(train_parse_args())
    print(params['labels_filename'])
    print(params['imgs_dir'])

    loaders = get_data_loaders(
        imgs_dir=params['imgs_dir'],
        labels_filename=params['labels_filename'],
        batch_size=params['batch_size'],
        n_imgs=params['n_imgs'])

    model = SimpleClassifier()
    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = BCELoss()
    trainer = Trainer(params, model, optimizer, criterion)

    trainer.run(loaders)


if __name__ == "__main__":
    main()

