# models.py

'''
CNN -> convolutional netwrok for  20-channels data

CNNOneChannel -> convolutional network for 1-channel images

all models output is just a linear function therefore:
- BCEWithlogit sounds fine for the loss function
- for the metrics entry final output needs sigmoid  

'''
import torch
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


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        out = self.outconv(dec1)
        return out
