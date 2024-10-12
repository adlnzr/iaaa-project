# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

'''
there are two main ViT concepts in the model architecture

1- VisionTransformerSPT_LSA 
input: iamges of each patient (sequence of patches) -> 
output: CLS of each image 

model: (ShiftedPatchTokenization + 12-layer transformer encoder layer + LocalitySelfAttention)

2- StandardViT
input: stacked CLS's of each patient (sequence of CLS's) ->  
output: CLS of each patient -> classifier

model: (12-layer transformer encoder layer)

'''

# +
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import ssl
from sklearn.utils.class_weight import compute_class_weight

from config import Config, Device
from datasets import MRIDataset, BalancedMRIDataset
from models import VisionTransformerSPT_LSA, VisionTransformerForTokens
from trainer import Trainer_ViT_smalldata
from tester import Tester_ViT_smalldata
# -

device = Device.device
print(device)

# +
# data_path = os.path.join(os.getcwd(), "data")
# labels_path = "train.csv"

# batch_size = Config.batch_size
# num_epochs = Config.num_epochs
# learning_rate = Config.learning_rate
# mean = Config.mean  # mean of the entire datasaet
# std = Config.std  # std of the entire dataaset
# image_size = 224

# +
data_path = os.path.join(os.getcwd(), "data")
labels_path = "train.csv"

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=0.5, std=0.5)
])

batch_size = 16

# +
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
# -

ssl._create_default_https_context = ssl._create_stdlib_context
model_1 = VisionTransformerSPT_LSA().to(device=device)

ssl._create_default_https_context = ssl._create_stdlib_context
model_2 = VisionTransformerForTokens().to(device=device)

model_name = model_1.__class__.__name__
model_name


# +
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
# -

criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = optim.AdamW(
    list(model_1.parameters()) + list(model_2.parameters()), 
    lr=3e-4, weight_decay=0.01)

# +
# # Dice Loss definition
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, inputs, targets):
#         inputs = torch.sigmoid(inputs)  # Apply sigmoid to ensure inputs are in range [0, 1]
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
#         return 1 - dice

# # Combined Dice + Weighted BCE Loss
# class CombinedLoss(nn.Module):
#     def __init__(self, class_weights=None, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
#         super(CombinedLoss, self).__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # Weighted BCE
#         self.dice_loss = DiceLoss(smooth=smooth)                 # Dice Loss
#         self.bce_weight = bce_weight
#         self.dice_weight = dice_weight

#     def forward(self, inputs, targets):
#         bce = self.bce_loss(inputs, targets)
#         dice = self.dice_loss(inputs, targets)
        
#         # Weighted combination of both losses
#         loss = self.bce_weight * bce + self.dice_weight * dice
#         return loss

# +

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



# +
trainer = Trainer_ViT_smalldata(
    model_1=model_1,
    model_2=model_2,
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
# -

model_1.load_state_dict(torch.load(f"saved_models/{model_1.__class__.__name__}_best.pth"))
model_2.load_state_dict(torch.load(f"saved_models/{model_2.__class__.__name__}_best.pth"))

# +
tester = Tester_ViT_smalldata(
    model_1 = model_1,
    model_2 = model_2,
    criterion=criterion,
    test_dl=test_dl,
    test_dataset=test_dataset,
    device=device,
    threshold=0.5
)

tester.test()
# -

torch.cuda.empty_cache()
