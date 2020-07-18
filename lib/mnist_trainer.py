import time

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from lib.models import CNNClassifier


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

    def log(self, item, name, epoch):
        #TODO other metrics
        self.writer.add_scalar(f"{name}/v{self.version}", item, epoch)

    def train_phase(self, model, optimizer, train_loader, epoch, device):
        model.train()
        result = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).double()

            loss = F.nll_loss(outputs, labels) #changed
            loss.backward()
            optimizer.step()
#            result += loss.item()

            if self.verbose:
                if batch_idx % 400 == 0:
                    print('train', epoch, batch_idx, loss.item())

        result /= 4000
        print('!', result)
        self.log(
            item=result,
            name='Train Loss',
            epoch=epoch)

    def valid_phase(self, model, valid_loader, epoch, device):
        model.eval()
        correct = 0
        for batch_idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            outputs = model(inputs).double()
            outputs_1 = torch.argmax(outputs, dim=0)
            outputs_2 = outputs_1.cpu().detach().numpy()
            outputs_3 = np.round(outputs_2)

            correct += outputs.cpu().eq(labels.data).cpu().sum()

            if batch_idx == 0:
                accuracy = []
            elif batch_idx % 100 == 0:
                result = sum(accuracy) / len(accuracy)
                self.log(
                    item=result,
                    name='Valid Accuracy',
                    epoch=epoch * len(valid_loader) + batch_idx)
                accuracy = []

            accuracy.append(accuracy_score(outputs_2, labels))

            if self.verbose:
                if batch_idx % 400 == 0:
                    print('------------')
                    print(labels)
                    print(outputs_1)
                    print(batch_idx, accuracy[-1])

        print("CORREST IS:")
        print(100. * correct / len(valid_loader.dataset))
        print("ACCURACY IS:")
        print(sum(accuracy) / len(accuracy), len(valid_loader.dataset))
        print("----")

    def run(self, loaders):
        if self.checkpoint_filename:
            epoch_start = self.load_from_checkpoint() # it loads state_dict for self.model and self.optimizer
        else:
            epoch_start = 1

        train_loader = loaders['train_loader']
        valid_loader = loaders['valid_loader']

        device = torch.device('cuda:1')
        model = CNNClassifier().to(self.params['device'])
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.5)
        self.valid_phase(model, valid_loader, 0, device)
        for epoch in range(epoch_start, epoch_start + self.n_epochs):
            if self.verbose:
                time_start = time.time()
                print(f"EPOCH {epoch}")

            self.train_phase(model, optimizer, train_loader, epoch, device)
            self.valid_phase(model, valid_loader, epoch, device)
            self.save_checkpoint(epoch)

            if self.verbose:
                print(f"Epoch time: {(time.time() - time_start) / 60} min")
