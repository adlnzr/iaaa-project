# config.py

import torch
import torch.nn as nn

class Device:
    # device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Config:
    # hyperparameters
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10

    # model parameters
    # input_size = 
    # hidden_size = 
    # output_size = 

    # criterion
    criterion = nn.BCEWithLogitsLoss()


