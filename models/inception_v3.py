# inception_v3.py

import torch.nn as nn
from torchvision.models import inception_v3

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()

        # Load the InceptionV3 model with pre-trained weights
        self.model = inception_v3(weights="IMAGENET1K_V1", transform_input=False)

        # Modify the first convolutional layer to accept grayscale input (1 channel)
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False
        )

        # Modify the fully connected layer to output 1 class (for binary classification)
        self.model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)

        # Initialize the weights of the modified layers
        nn.init.kaiming_normal_(self.model.Conv2d_1a_3x3.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)

        # Make all parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)
