import os
import random
import pandas as pd
import numpy as np
import torch
import pydicom
from PIL import Image
from torch.utils.data import Dataset


class InferenceDataset(Dataset):
    def __init__(self, data_path, transform=None, augment_transform=None, max_slices=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.max_slices = max_slices

        self.patient_list = [i.name for i in data_path.iterdir() if i.is_dir()]

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        patient_id = self.patient_list[idx]
        images = self.load_patient_images(patient_id)

        # data needs to be "array"/"PIL image" to be applied by transforms
        images = [Image.fromarray(image, mode='L') if isinstance(
            image, np.ndarray) else image for image in images]

        if self.transform:
            images = [self.transform(image) for image in images]
        else:
            images = [torch.tensor(img, dtype=torch.float32) for img in images]

        images = torch.stack(images)
        images = images.squeeze(1)  # remove the channel dimension

        return images

    def load_patient_images(self, patient_id):
        patient_dir = os.path.join(self.data_path, patient_id)
        dcm_paths = [os.path.join(patient_dir, f) for f in os.listdir(
            patient_dir) if f.endswith('.dcm')]

        images = []
        for dcm_path in dcm_paths:
            dcm_data = pydicom.dcmread(dcm_path)
            image_array = dcm_data.pixel_array
            # image_array = image_array / np.max(image_array)  # normalize the image
            images.append(image_array)

        # pad the images to max_slices
        if self.max_slices:
            padding_needed = self.max_slices - len(images)
            if padding_needed > 0:
                extra_slices = random.choices(images, k=padding_needed)
                images.extend(extra_slices)

        return images
