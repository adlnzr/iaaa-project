"""
python submission.py --data-dir /path/to/data-dir --predictions-file-path /path/to/submission.csv
"""

import os
import ssl
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config, Device
from datasets import InferenceDataset
from models import ResnextViT
from inference import inference


HERE = Path(__file__).absolute().resolve().parent


@click.command()
@click.option(
    "--data-dir",
    type=Path,
    help="path to data directory, which consists of folders of Dicom files, each one corresponding to a Dicom series.",
)
@click.option("--predictions-file-path", type=Path)
def main(data_dir: Path, predictions_file_path: Path):
    device = Device.device
    data_path = data_dir

    batch_size = Config.batch_size
    mean = Config.mean
    std = Config.std
    image_size = 224

    resclaed_mean = round(mean/255, 4)
    rescaled_std = round(std/255, 4)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[resclaed_mean], std=[rescaled_std])
    ])

    series_instance_uid_list = [
        i.name for i in data_dir.iterdir() if i.is_dir()]

    inference_dataset = InferenceDataset(
        data_dir,
        transform=transform,
        max_slices=20
    )

    inference_dl = DataLoader(inference_dataset, batch_size=32)

    model = ResnextViT(pretrained=False).to(device)

    model_name = model.__class__.__name__

    model.load_state_dict(torch.load(
        f"saved_models/{model_name}.pth", map_location=torch.device(device=device)))


    predictions = inference(model, inference_dl, device)

    predictions_df = pd.DataFrame(
        {
            "SeriesInstanceUID": series_instance_uid_list,
            "prediction": predictions,
        }
    )
    predictions_df.to_csv(predictions_file_path, index=False)


if __name__ == "__main__":
    main()
