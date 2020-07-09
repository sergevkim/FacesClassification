import time

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize


class Trainer:

    def __init__(self, params, model, optimizer, criterion):
        self.params = params
        self.checkpoints_dir = self.params['checkpoints_dir']
        self.checkpoint_filename = self.params['checkpoint_filename']
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

    def save_checkpoint(self, epoch):
        checkpoint = {
            'model': self.model,
            'optimizer': self.optimizer,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        checkpoints_dir = self.params['checkpoints_dir'],
        checkpoint_path = f"{self.checkpoints_dir}/v{self.version}-e{epoch}.hdf5"
        torch.save(checkpoint, checkpoint_path)

    def load_from_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model = checkpoint['model'].to(self.device)
        self.optimizer = checkpoint['optimizer'].to(self.device) # ask about these two lines of code
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']

        return epoch_start

    def log(self, item, name, epoch):
        #TODO other metrics
        self.writer.add_scalar(f"{name}/v{self.version}", item, epoch)

    def train_phase(self, train_loader, epoch):
        self.model.train()
        result = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs).double()

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels) #changed
            result += loss.item()
            loss.backward()
            self.optimizer.step()

            if self.verbose:
                if batch_idx % 100 == 0:
                    print('train', epoch, batch_idx, loss.item())

        result /= 4000
        print('!', result)
        self.log(
            item=result,
            name='Train Loss',
            epoch=epoch)

    def valid_phase(self, valid_loader, epoch):
        self.model.eval()

        for batch_idx, batch in enumerate(valid_loader):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            #labels = labels.to(self.device)
            outputs = self.model(inputs).double()
            outputs_1 = torch.argmax(outputs, dim=0)
            outputs_2 = outputs_1.cpu().detach().numpy()
            outputs_3 = np.round(outputs_2)

            if batch_idx == 0:
                accuracy = []
            elif batch_idx % 100 == 0:
                result = sum(accuracy) / len(accuracy)
                self.log(
                    item=result,
                    name='Valid Accuracy',
                    epoch=epoch * len(valid_loader) + batch_idx)
                accuracy = []

            accuracy.append(accuracy_score(outputs_3, labels))

            if self.verbose:
                if batch_idx % 100 == 0:
                    print(batch_idx, accuracy[-1])
                    
        print(sum(accuracy) / len(accuracy), len(valid_loader))

    def run(self, loaders):
        if self.checkpoint_filename:
            epoch_start = self.load_from_checkpoint() # it loads state_dict for self.model and self.optimizer
        else:
            epoch_start = 1

        train_loader = loaders['train_loader']
        valid_loader = loaders['valid_loader']

        self.valid_phase(valid_loader, epoch=0)
        for epoch in range(epoch_start, epoch_start + self.n_epochs):
            if self.verbose:
                time_start = time.time()
                print(f"EPOCH {epoch}")

            self.train_phase(train_loader, epoch)
            self.valid_phase(valid_loader, epoch)
            self.save_checkpoint(epoch)

            if self.verbose:
                print(f"Epoch time: {(time.time() - time_start) / 60} min")
