import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from lib.models import SimpleClassifier
from lib.utils import get_data_loaders, train_parse_args


class Trainer:
    def __init__(self, mode='.py', params, model=None, optimizer=None, criterion=None):
        self.params = params
        self.imgs_dir = self.params['imgs_dir']
        self.labels_filename = self.params['labels_filename']
        self.checkpoints_dir = self.params['checkpoints_dir']
        self.logs_dir = self.params['logs_dir']
        self.version = self.params['version']
        self.verbose = self.params['verbose']

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = SummaryWriter(self.logs_dir)

    def save(self, epoch):
        state_dict = {
            'model': self.model,
            'optimizer': self.optimizer,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        checkpoints_dir = self.params['checkpoints_dir'],
        checkpoint_path = f"{self.checkpoints_dir}/v{self.version}-e{epoch}.hdf5"
        torch.save(state_dict, checkpoint_path)

    def log(self, accuracy, epoch):
        #TODO other metrics
        self.writer.add_scalar('accuracy', accuracy.item(), epoch)

    def train_phase(self, train_loader, epoch):
        for batch_idx, batch in enumerate(train_loader):
            self.model.train()
            self.optimizer.zero_grad()

            inputs, labels = batch
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def valid_phase(self, val_loader, epoch):
        self.model.val()

        accuracy = None #TODO

        self.save(epoch)
        self.log(accuracy, epoch)

    def run(self, loaders):
        train_loader = loaders['train_loader']
        valid_loader = loaders['valid_loader']

        for epoch in range(params['n_epochs']):
            if self.verbose:
                print("EPOCH {}".format(epoch))

            self.train_phase(train_loader, epoch)
            self.valid_phase(valid_loader, epoch)


def main():
    params = vars(train_parse_args())

    loaders = get_data_loaders(
        imgs_dir=params['imgs_dir'],
        labels_filename=params['labels_filename'],
        batch_size=params['batch_size'])

    model = SimpleClassifier()
    optimizer = Adam(model.parameters(), lr=3e-4)
    criterion = BCELoss()
    trainer = Trainer(params, model, optimizer, criterion)

    trainer.run(loaders)


if __name__ == "__main__":
    main()

