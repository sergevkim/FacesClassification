import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def get_resnet(pretrained=False):
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model


class ResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet, self).__init__()
        self.backbone = resnet18(pretrained=pretrained)
        #self.head_bald = nn.Linear(self.backbone.fc.out_features, 1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, inputs):
        x = self.backbone(inputs)
        #x = self.head_bald(x)

        return torch.sigmoid(x)

    
class SimpleClassifierMNIST(nn.Module):
    def __init__(self):
        super(SimpleClassifierMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1)

        self.fc1 = nn.Linear(8 * 7 * 7, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.nn.Softmax()(x)


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1)

        self.fc1 = nn.Linear(8 * 128 * 128, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, (2, 2))
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, (2, 2))
        
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)
