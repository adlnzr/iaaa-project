# mri_dataset.py

import os
import random
import pandas as pd
import numpy as np
import torch
import pydicom
from PIL import Image
from torch.utils.data import Dataset

class MRIDataset(Dataset):  # no augmentation / original unbalanced data

    '''
    output -> image , label 
    image: extracted from pydicom file of each patient and then transformed
    output image dimension: [1,h,w]
    '''

    def __init__(self, data_path, labels_path, split="train", transform=None, random_state=42):
        super().__init__()
        self.data_path = data_path
        self.labels_path = labels_path
        self.transform = transform

        self.labels = pd.read_csv(labels_path)

        normal_path = list(self.labels[self.labels['prediction'] == 0].reset_index(
            drop=True)["SeriesInstanceUID"])
        abnormal_path = list(self.labels[self.labels['prediction'] == 1].reset_index(
            drop=True)["SeriesInstanceUID"])

        random.Random(random_state).shuffle(normal_path)
        random.Random(random_state).shuffle(abnormal_path)

        normal_val_samples = int(len(normal_path) * 0.2)
        abnormal_val_samples = int(len(abnormal_path) * 0.2)

        if split == "train":
            self.patient_list = normal_path[normal_val_samples *
                                            2:] + abnormal_path[abnormal_val_samples*2:]
        elif split == "val":
            self.patient_list = normal_path[:normal_val_samples] + \
                abnormal_path[:abnormal_val_samples]
        elif split == "test":
            self.patient_list = normal_path[normal_val_samples: normal_val_samples *
                                            2] + abnormal_path[abnormal_val_samples: abnormal_val_samples*2]

        random.Random(random_state).shuffle(self.patient_list)

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        patient_id = self.patient_list[idx]
        image = self.load_patient_images(patient_id)

        # data needs to be "array"/"PIL image" to be applied by transforms
        image = [Image.fromarray(image, mode= 'L') if isinstance(image, np.ndarray) else image]
                
        if self.transform:
            image = self.transform(image)
        else:
            images = torch.tensor(image, dtype=torch.float32)

        # dimension: [h,w]? to be checked
        images = image.unsqueeze(0)  # [1,h,w] adding the channel dimension

        label = self.labels[self.labels['SeriesInstanceUID']
                            == patient_id]['prediction'].values
        label = torch.tensor(label, dtype=torch.long)

        return images, label

    def load_patient_images(self, patient_id):
        patient_dir = os.path.join(self.data_path, patient_id)
        dcm_paths = [os.path.join(patient_dir, f) for f in os.listdir(
            patient_dir) if f.endswith('.dcm')]

        for dcm_path in dcm_paths:
            dcm_data = pydicom.dcmread(dcm_path)
            image_array = dcm_data.pixel_array
            # image_array = image_array / \
            #     np.max(image_array)  # normalize the image
            return image_array
