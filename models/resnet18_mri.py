# resnet18_mri.py

import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

class ResNet18MRI(nn.Module):
    def __init__(self):
        super(ResNet18MRI, self).__init__()

        # Load the processor and model pre-trained on MRI brain scans
        self.processor = AutoImageProcessor.from_pretrained("BehradG/resnet-18-finetuned-MRI-Brain")
        self.model = AutoModelForImageClassification.from_pretrained(
            "BehradG/resnet-18-finetuned-MRI-Brain", 
            num_channels=1,  # Configuring to accept grayscale images
            ignore_mismatched_sizes=True  # Allows loading model with different layer sizes
        )

        # Initialize the weights using Kaiming initialization for the convolution layer
        nn.init.kaiming_normal_(
            self.model.resnet.embedder.embedder.convolution.weight, 
            mode='fan_out', 
            nonlinearity='relu'
        )

        # Initialize the bias to zeros if present
        if self.model.resnet.embedder.embedder.convolution.bias is not None:
            nn.init.zeros_(self.model.resnet.embedder.embedder.convolution.bias)
        
        # Modify the last fully connected layer to output 1 class (for binary classification)
        self.linear_layer = self.model.classifier[1]
        self.modified_linear_layer = nn.Linear(
            in_features=self.linear_layer.in_features, 
            out_features=1
        )
        self.model.classifier[1] = self.modified_linear_layer

        # Make all parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Forward pass through the model
        x = self.model(x)
        return x
