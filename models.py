# models.py

'''
CNN -> convolutional netwrok for  20-channels data

CNNOneChannel -> convolutional network for 1-channel images

all models output is just a linear function therefore:
- BCEWithlogit sounds fine for the loss function
- for the metrics entry final output needs sigmoid  

'''
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNN(nn.Module):
    def __init__(self, in_channels=20, out_channels=1):
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

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = self.maxpool(x)
        x = F.relu(self.cnn2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNOneChannel(nn.Module):
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

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = self.maxpool(x)
        x = F.relu(self.cnn2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class MriResentModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv = nn.Conv2d(in_dim, 3, kernel_size=1, padding=0)

        # instantiate transfer learning model
        self.resnet_model = models.resnet50(pretrained=True)

        # set all paramters as trainable
        for param in self.resnet_model.parameters():
            param.requires_grad = True

        # get input of fc layer
        n_inputs = self.resnet_model.fc.in_features  # 2048

        # redefine fc layer / top layer/ head for our classification problem
        self.resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                             nn.SELU(),
                                             nn.Dropout(p=0.4),
                                             nn.Linear(2048, 2048),
                                             nn.SELU(),
                                             nn.Dropout(p=0.4),
                                             nn.Linear(2048, out_dim),
                                            #  nn.LogSigmoid()
        )

        # set model to run on GPU or CPU absed on availibility
        # self.resnet_model.to(device)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet_model(x)
        return x
