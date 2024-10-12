#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ssl

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from config import Config, Device
from datasets import MRIDataset, BalancedMRIDataset
from models import UNetResnet
from trainer import Trainer
from tester import Tester


# In[2]:


device = Device.device
print(device)


# In[3]:


data_path = os.path.join(os.getcwd(), "data")
labels_path = "train.csv"

batch_size = Config.batch_size
num_epochs = Config.num_epochs
learning_rate = Config.learning_rate
mean = Config.mean # mean of the entire datasaet
std = Config.std # std of the entire dataaset
image_size = 224


# In[4]:


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=0.5, std=0.5)
])


# In[5]:


train_dataset = MRIDataset(
    data_path,
    labels_path,
    split='train',
    transform=transforms,
    max_slices=20
)

val_dataset = MRIDataset(
    data_path,
    labels_path,
    split='val',
    transform=transforms,
    max_slices=20
)

test_dataset = MRIDataset(
    data_path,
    labels_path,
    split='test',
    transform=transforms,
    max_slices=20
)

train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=16)
test_dl = DataLoader(test_dataset, batch_size=16)


# In[6]:


ssl._create_default_https_context = ssl._create_stdlib_context


# In[7]:


model = UNetResnet().to(device=device)


# In[8]:


model_name = model.__class__.__name__
model_name


# In[9]:


def compute_class_weights_from_csv(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    labels = df['prediction'].values

    # Convert labels to integers if they are not already
    labels = labels.astype(int)

    # Compute class weights
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced', classes=unique_labels, y=labels)

    # Convert to torch tensor
    return torch.tensor(class_weights, dtype=torch.float)


# Path to your CSV file
class_weights = compute_class_weights_from_csv(labels_path)

# For binary classification, use the appropriate class weight
# Assuming binary classification with class labels 0 and 1
class_weights = class_weights[1]  # Adjust if necessary
print("Class Weights:", class_weights)


# In[10]:


# Dice Loss definition
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to ensure inputs are in range [0, 1]
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

# Combined Dice + Weighted BCE Loss
class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # Weighted BCE
        self.dice_loss = DiceLoss(smooth=smooth)                 # Dice Loss
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        # Weighted combination of both losses
        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss


# In[11]:


criterion = CombinedLoss(class_weights=class_weights).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


# In[12]:


trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_dl=train_dl,
    val_dl=val_dl,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    num_epochs=80,
    patience=20,
    threshold=0.5,
    save_path=f"saved_models/{model_name}.pth"
)

# Start training
trainer.train()


# In[13]:


model.load_state_dict(torch.load(f"saved_models/{model.__class__.__name__}_best.pth"))


# In[14]:


tester = Tester(
    model=model,
    criterion = criterion,
    test_dl=test_dl,
    test_dataset=test_dataset,
    device=device,
    threshold=0  # Set the threshold for binary classification
)

# Perform testing and print metrics
tester.test()

