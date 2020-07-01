import time

import torch
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, params, model, optimizer, criterion):
        self.params = params
        self.checkpoints_dir = self.params['checkpoints_dir']
        self.device = self.params['device']
        self.imgs_dir = self.params['imgs_dir']
        self.labels_filename = self.params['labels_filename']
        self.logs_dir = self.params['logs_dir']
        self.n_epochs = self.params['n_epochs']
        self.verbose = self.params['verbose']
        self.version = self.params['version']

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
        self.model.train()

        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs).double()

            #TODO wtf
            #outputs = outputs.view_as(labels)

            loss = self.criterion(outputs, labels.unsqueeze(1))

            if self.verbose:
                if batch_idx % 50 == 0:
                    print(epoch, batch_idx, loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def valid_phase(self, valid_loader, epoch):
        self.model.eval()

        accuracy = []

        for batch_idx, batch in enumerate(valid_loader):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs).double()
            outputs_2 = outputs.cpu().detach().numpy().round() * 2 - 1
            labels_2 = labels.cpu().detach().numpy()

            accuracy.append(accuracy_score(outputs_2, labels_2))

            if self.verbose:
                if batch_idx % 50 == 0:
                    print(batch_idx, accuracy[-1])

        result = sum(accuracy) / len(accuracy)
        self.log(result, epoch)

    def run(self, loaders):
        train_loader = loaders['train_loader']
        valid_loader = loaders['valid_loader']

        self.valid_phase(valid_loader, epoch=0)
        for epoch in range(1, self.n_epochs + 1):
            if self.verbose:
                time_start = time.time()
                print("EPOCH {}".format(epoch))

            self.train_phase(train_loader, epoch)
            self.valid_phase(valid_loader, epoch)
            self.save(epoch)

            if self.verbose:
                print(f"Epoch time: {time_start - time.time()}")

