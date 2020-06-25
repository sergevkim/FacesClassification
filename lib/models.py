import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.main = Sequential()

    def forward(self, x):
        pass

