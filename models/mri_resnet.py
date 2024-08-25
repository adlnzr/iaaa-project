# mri_resnet.py

import torch.nn as nn
from torchvision import models

class MriResentModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.resnet_model = models.resnet50(pretrained=True)

        # modify the first convolutional layer to accept 1-channel input 
        self.resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # reinitialize the weights of this layer
        nn.init.kaiming_normal_(self.resnet_model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # get input of fc layer (last layer)
        n_inputs = self.resnet_model.fc.in_features  # 2048

        # redefine fc layer / top layer/ head for our classification problem
        self.resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                             nn.SELU(),
                                             nn.Dropout(p=0.4),
                                             nn.Linear(2048, 2048),
                                             nn.SELU(),
                                             nn.Dropout(p=0.4),
                                             nn.Linear(2048, out_dim)                  
        ) # Note - no softmax on last layer so sigmoid/softmax for the criterion and metrics
        
        # set all paramters as trainable
        for param in self.resnet_model.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.resnet_model(x)
        return x