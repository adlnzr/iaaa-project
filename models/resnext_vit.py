import torch
from transformers import ViTModel
from torchvision import models
from torch import nn


class ResnextViT(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k', num_classes=1):
        super(ResnextViT, self).__init__()

        # Load a pre-trained ResNeXt model and remove the final FC layer and pooling
        self.resnext = models.resnext50_32x4d(pretrained=True)

        # Modify the first convolutional layer to accept 1 channel instead of 3
        self.resnext.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize the new convolutional layer with random weights
        nn.init.kaiming_normal_(self.resnext.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Remove final FC layer and pooling
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-2])  

        # Load a pre-trained ViT model
        self.vit = ViTModel.from_pretrained(vit_model_name)
        
        # Project ResNeXt features from 2048 to 768 for ViT
        self.feature_projection = nn.Linear(2048, self.vit.config.hidden_size)  
        
        # Binary classification head
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        
        # Positional embeddings
        self.positional_embeddings = self.vit.embeddings.position_embeddings # torch.Size([1, 197, 768])

    def forward(self, x):
        # x shape [batch_size, 1, 224, 224]
        batch_size = x.shape[0]

        # extract features from ResNeXt
        x = self.resnext(x)  # shape: [batch_size, 2048, 7, 7]

        # flatten the ResNeXt feature map: [batch_size, 2048, 7, 7] -> [batch_size, 49, 2048]
        x = x.flatten(2).transpose(1, 2)  # now x is [batch_size, 49, 2048]
        
        # project the ResNeXt features to match ViT's embedding dimension (768)
        x = self.feature_projection(x)  # now x is [batch_size, 49, 768]
        
        # add the CLS token
        cls_token = torch.zeros(batch_size, 1, x.size(-1), device=x.device)  # shape: [batch_size, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # Now x is [batch_size, 50, 768] (49 patches + 1 CLS token)
        
        pos_embeddings = self.positional_embeddings[:, :x.size(1), :]  # shape: [1, 50, 768]
        x += pos_embeddings # pytorch broadcasts pos_embeddings along the batch dimension
                            
        # pass through the ViT encoder (bypassing the patch embedding step)
        outputs = self.vit.encoder(x)
        
        # extract the [CLS] token output for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        logits = self.classifier(cls_output)
        
        return logits