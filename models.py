# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoConfig, AutoModelForImageClassification, AutoImageProcessor


'''
CNN class ->

convolutional netwrok, input: 1-channel matrix to nn.Conv2d, 
                        output: 1-dim/scaler from nn.Linear
initialization applied for all conv and linear layers
'''

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

'''

'''

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

'''
class ResNet18MRI ->

resnet 18 pretrained model on brain mri images
model first layer modified to take 1-channel image instead of 3-channel
model last layer modified to output=1 instead of output=2
model fine tuned on just first layer and last layer 
'''

class ResNet18MRI(nn.Module):
    def __init__(self):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained("BehradG/resnet-18-finetuned-MRI-Brain")
        self.model = AutoModelForImageClassification.from_pretrained("BehradG/resnet-18-finetuned-MRI-Brain", 
                                                         num_channels=1,
                                                         ignore_mismatched_sizes=True)

        # initialize the weights using Kaiming initialization
        nn.init.kaiming_normal_(self.model.resnet.embedder.embedder.convolution.weight, mode='fan_out', nonlinearity='relu')

        if self.model.resnet.embedder.embedder.convolution.bias is not None:
            nn.init.zeros_(self.model.resnet.embedder.embedder.convolution.bias)
        
        # modifying last fully connected layer to change the outputs=1
        self.linear_layer = self.model.classifier[1]
        self.modified_linear_layer = nn.Linear(in_features=self.linear_layer.in_features, out_features=1)
        self.model.classifier[1] = self.modified_linear_layer

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.resnet.embedder.embedder.convolution.parameters():
            param.requires_grad = True

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.model(x)
        return x


# class CNN(nn.Module):
#     def __init__(self, in_channels=20, out_channels=1):
#         super().__init__()

#         self.cnn1 = nn.Conv2d(in_channels=in_channels,
#                               out_channels=64, kernel_size=5)
#         # size -> (64,220,220)
#         self.maxpool = nn.MaxPool2d(kernel_size=2)
#         # size -> (64,110,110)
#         self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
#         # size -> (128,106,106)
#         # maxpool -> (128,53,53)
#         self.fc1 = nn.Linear(128*53*53, 300)
#         self.fc2 = nn.Linear(300, 1)

#     def forward(self, x):
#         x = F.relu(self.cnn1(x))
#         x = self.maxpool(x)
#         x = F.relu(self.cnn2(x))
#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)  # flatten

#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x