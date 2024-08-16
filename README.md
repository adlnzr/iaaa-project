# MRI Image Analysis Project

## Project Overview
This project is focused on developing and experimenting with various deep learning models for analyzing MRI images. The models are primarily aimed at medical diagnostics, where MRI images are classified into specific categories.

## Key Components

1. **Models**:
   - **CNN**: Convolutional neural network for processing multi-channel MRI images.
   - **CNNOneChannel**: A variant of CNN for single-channel images.
   - **ResNet-50**: A modified ResNet-50 model for MRI data.
   - **UNet**: Used for image segmentation.

2. **Datasets**:
   - **MRIDataset**: A standard dataset with an imbalanced class distribution (20:80).
   - **BalancedMRIDataset**: An augmented dataset to balance the class distribution (42:58).

3. **Training and Testing**:
   - **train.py**: Implements training routines for both multi-channel and single-channel datasets, with early stopping based on evaluation metrics.
   - **test.py**: Evaluates the trained models, providing metrics like accuracy, precision, recall, and AUC.

4. **Configuration**:
   - Configurations such as learning rate, batch size, and number of epochs are defined in `config.py`.

5. **Data Handling**:
   - **datasets.py** handles loading, preprocessing, and augmenting MRI images.

6. **Evaluation Metrics**:
   - Precision, recall, ROC-AUC score, and average metrics are used to assess model performance.

## Usage
1. **Setup**: Configure paths and parameters in `config.py`.
2. **Training**: Run `train.py` to train the models.
3. **Testing**: Use `test.py` to evaluate model performance on the test dataset.

## Files in the Project
- **`cnn.ipynb`**: Contains CNN implementation.
- **`cnn_one_channel.ipynb`**: Single-channel CNN experiments.
- **`resnet_50.ipynb`**: Experiments with the ResNet-50 model.
- **`u-net.ipynb`**: U-Net implementation and experiments.
- **`train.py`**: Script for training models.
- **`test.py`**: Script for testing and evaluation.
- **`config.py`**: Configuration settings.
- **`datasets.py`**: Dataset class definitions and data handling logic.
- **`train.csv`**: Training data annotations.

This README provides an overview and instructions for using the provided models and scripts.
