# config.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pydicom


class Device:
    # device configuration
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class Config:
    # hyperparameters
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10
    mean = 65.27065710775146
    std = 62.37104934969386

    # model parameters
    # input_size = 
    # hidden_size = 
    # output_size = 

    # criterion
    criterion = nn.BCEWithLogitsLoss()

'''
class Summary: to compute mean and std of the dataset
it takes a while to compute each time imported so it is commented out

the mean and std have been calculated once in data_summary.ipynb and 
those values are in class Config for easier access 
'''
# class Summary:

#     data_path = os.path.join(os.getcwd(), "data")

#     total_sum = 0.0
#     total_squared_sum = 0.0
#     total_pixels = 0
#     total_files = 0

#     # iterate over each patient directory
#     for patient_id in os.listdir(data_path):
#         patient_path = os.path.join(data_path, patient_id)
        
#         # ensure we're only working with directories
#         if os.path.isdir(patient_path):
#             # iterate over each DICOM file in the patient's directory
#             for dcm_file in os.listdir(patient_path):
#                 dcm_path = os.path.join(patient_path, dcm_file)
                
#                 # ensure it's a file
#                 if os.path.isfile(dcm_path):
                    
#                     dicom_file = pydicom.dcmread(dcm_path)
#                     image_array = dicom_file.pixel_array
                    
#                     total_sum += np.sum(image_array)
#                     total_squared_sum += np.sum(image_array ** 2)
#                     total_pixels += image_array.size
#                     total_files += 1

#     mean = total_sum / total_pixels
#     variance = (total_squared_sum / total_pixels) - (mean ** 2)
#     std = np.sqrt(variance)


