#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
model architecture: resnext + ViT + classifier(linear) -> kinda a hybrid approach

ResNext for Low-level Features: ResNet, particularly the initial layers, extracting 
low-level features (edges, textures, patterns) from images
ViT for Global Attention: capturing long-range dependencies and global context through 
its self-attention mechanism.

just the [CLS] token of the last layer of the ViT goes to the classifier

'''


# In[3]:


import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from config import Config, Device
from datasets import MRIDataset, BalancedMRIDataset
from models import ResnextViT
from trainer import Trainer
from tester import Tester


# In[4]:


device = Device.device
print(device)


# In[5]:


data_path = os.path.join(os.getcwd(), "data")
labels_path = "train.csv"

batch_size = Config.batch_size
num_epochs = Config.num_epochs
learning_rate = Config.learning_rate
mean = Config.mean # mean of the entire datasaet
std = Config.std # std of the entire dataaset
image_size = 224


# In[6]:


resclaed_mean = round(mean/255,4) # re-scale the actual mean
rescaled_std = round(std/255, 4) # re-scale the actual std

train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[resclaed_mean], std=[rescaled_std])
])

augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[resclaed_mean], std=[rescaled_std])
])

test_transforms = transforms.Compose([
    # transforms.Lambda(lambda img: img.astype(np.float32)),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[resclaed_mean], std=[rescaled_std])
])


# In[7]:


train_dataset = BalancedMRIDataset(
    data_path,
    labels_path,
    split='train',
    transform=train_transforms,
    augment_transform=augment_transforms,
    augment=True,
    max_slices=20
)

val_dataset = BalancedMRIDataset(
    data_path,
    labels_path,
    split='val',
    transform=test_transforms,
    max_slices=20
)

test_dataset = BalancedMRIDataset(
    data_path,
    labels_path,
    split='test',
    transform=test_transforms,
    max_slices=20
)

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32)
test_dl = DataLoader(test_dataset, batch_size=32)


# In[8]:


model = ResnextViT().to(device)
# model = ResnextViT()
model


# In[9]:


model.to(device=device)


# In[10]:


model_name = model.__class__.__name__
model_name


# In[11]:


# loss and optimizer

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


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
    num_epochs=70,
    patience=100,
    threshold=0.5,
    save_path=f"saved_models/{model_name}.pth"
)

# Start training
trainer.train()


# In[13]:


# model.load_state_dict(torch.load(f"saved_models/{model_name}.pth"))


# In[14]:


# tester = Tester(
#     model=model,
#     criterion=criterion,
#     test_dl=test_dl,
#     test_dataset=test_dataset,
#     device=device,
#     threshold=0.5
# )

# tester.test()


# In[15]:


# !tensorboard --logdir=runs/training_AutoencoderClassifier


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


'''
CustomViT(
  (vit): ViTModel(
    (embeddings): ViTEmbeddings(
      (patch_embeddings): ViTPatchEmbeddings(
        (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): ViTEncoder(
      (layer): ModuleList(
        (0-11): 12 x ViTLayer(
          (attention): ViTSdpaAttention(
            (attention): ViTSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): ViTSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): ViTIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): ViTOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (pooler): ViTPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (original_embeddings): ViTEmbeddings(
    (patch_embeddings): ViTPatchEmbeddings(
      (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (dropout): Dropout(p=0.0, inplace=False)
  )
)
'''

