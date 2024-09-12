# unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_mri import ResNet18MRI


class UNetResnet(nn.Module):
    def __init__(self):
        super(UNetResnet, self).__init__()

        self.unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=3, out_channels=1, init_features=32, pretrained=True)
        
        self.unet.encoder1.enc1conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.kaiming_normal_(self.unet.encoder1.enc1conv1.weight, mode='fan_out', nonlinearity='relu')

        self.resnet = ResNet18MRI()

        for param in self.unet.parameters():
            param.requires_grad = True

        for param in self.resnet.parameters():
            param.requires_grad = True


    def forward(self, x):
        x = self.unet(x)
        x = self.resnet(x)
        return x
