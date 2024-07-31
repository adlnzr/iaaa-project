import os
import random

import pandas as pd
import numpy as np
import torch
import pydicom

from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, data_path, labels_path, split="train", transform=None, max_slices=None, random_state=42):
        super().__init__()
        self.data_path = data_path
        self.labels_path = labels_path
        self.transform = transform
        self.max_slices = max_slices

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
        images = self.load_patient_images(patient_id)

        if self.transform:
            images = [self.transform(image) for image in images]
        else:
            images = [torch.tensor(img, dtype=torch.float32) for img in images]

        # [18,1,h,w] stack images along the 0th dimension
        images = torch.stack(images)
        images = images.squeeze(1)  # [18,h,w] remove the channel dimension

        label = self.labels[self.labels['SeriesInstanceUID']
                            == patient_id]['prediction'].values
        label = torch.tensor(label, dtype=torch.long)

        return images, label

    def load_patient_images(self, patient_id):

        patient_dir = os.path.join(self.data_path, patient_id)
        dcm_paths = [os.path.join(patient_dir, f) for f in os.listdir(
            patient_dir) if f.endswith('.dcm')]

        images = []
        for dcm_path in dcm_paths:
            dcm_data = pydicom.dcmread(dcm_path)
            image_array = dcm_data.pixel_array
            image_array = image_array / \
                np.max(image_array)  # normalize the image
            images.append(image_array)

        # pad the images to max_slices
        if self.max_slices:
            padding_needed = self.max_slices - len(images)
            if padding_needed > 0:
                padding_image = np.zeros_like(images[0])
                images.extend([padding_image] * padding_needed)

        return images
