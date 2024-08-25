# cnn.py

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=64, kernel_size=5)
        # size-> (64,220,220)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # size -> (64,110,110)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        # size -> (128,106,106)
        # maxpool -> (128,53,53)
        self.fc1 = nn.Linear(128*53*53, 300)
        self.fc2 = nn.Linear(300, 1)

        # apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = self.maxpool(x)
        x = F.relu(self.cnn2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _initialize_weights(self):
        # initialize weights of convolutional layers
        nn.init.kaiming_normal_(self.cnn1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.cnn2.weight, mode='fan_out', nonlinearity='relu')
        
        # initialize weights of fully connected layers
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        
        # initialize biases with zeros
        nn.init.constant_(self.cnn1.bias, 0)
        nn.init.constant_(self.cnn2.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
