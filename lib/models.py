import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(3 * 256 * 256, 120)
        self.fc3 = nn.Linear(120, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = self.fc3(x)

        return nn.Softmax()(x)

