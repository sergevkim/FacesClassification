import numpy as np

import torch
import torch.nn as nn
from torch.nn import Conv2d, Dropout, Linear, ReLU, Sigmoid
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
        self.backbone.fc = Linear(self.backbone.fc.in_features, 120)
        self.head = Linear(120, 1)

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = ReLU()(x)
        x = Dropout()(x)
        x = self.head(x)

        return Sigmoid()(x)
    
    def predict(self, inputs):
        outputs = self.forward(inputs)
        predict = outputs.round()

        return predict, outputs

    
class CNNClassifier(nn.Module):
    """Custom module for a simple convnet classifier"""
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        
        x = x.view(-1, 320)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        
        # transform to logits
        x = F.log_softmax(x)
        
        return x


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.conv4 = Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.conv5 = Conv2d(32, 16, kernel_size=(3, 3), padding=1)
        self.conv6 = Conv2d(16, 8, kernel_size=(3, 3), padding=1)

        self.fc1 = Linear(8 * 218 * 178, 120)
        self.fc2 = Linear(120, 1)

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

        return x
    
    def predict(self, x):
        outputs = self.forward(x)
        pred = outputs.round()

        return pred
