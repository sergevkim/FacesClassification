import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib


class SimpleClassifier(nn.Module):


    def __init__(self):
        super(SimpleClassifier, self).__init__()


    def forward(self, x):
        pass


    def training_step(self, batch):
        pass


    def validation_step(self, batch):
        pass


    def configure_optimizers(self):
        #TODO find_lr
        return torch.optim.Adam(self.parameters(), lr=3e-4)


    def train_dataloader(self):
        pass


    def val_dataloader(self):
        pass


