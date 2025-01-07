# Indoor Panorama Semantic Segmentation 

## Overview
Developed a deep learning model (using UNet architecture) for semantic segmentation of indoor panoramic images using PyTorch.

## Key Features

### Data Augmentation
- Image Resizing: 100x720 pixels
- Transformations:
  - Random Rotation (±35 degrees)
  - Horizontal Flip (50% probability)
  - Vertical Flip (10% probability)
  - Normalization

### Model Configuration
- **Architecture**: Custom UNET
- **Training Parameters**:
  - Loss Function: Mean Squared Error (MSELoss)
  - Optimizer: Adam
  - Learning Rate: 1e-4
  - Epochs: 100
  - Batch Size: 1

## Installation
```bash
# Clone repository
git clone <repository-url>

# Install dependencies
pip install torch albumentations numpy matplotlib opencv-python
```

## Usage
```python
# Import required modules
from src.training import train_model
from src.dataset_2 import SegmentationDataset

# Initialize dataset and train model
dataset = SegmentationDataset(image_dir, mask_dir)
train_model(dataset)
```

## Training Process
1. Load images and masks
2. Apply data augmentation
3. Normalize pixel values
4. Train UNET model
5. Save checkpoints periodically

## Visualization
- Displays predicted vs ground truth masks
- Saves prediction images during evaluation

## Requirements
- PyTorch
- Albumentations
- NumPy
- Matplotlib
- OpenCV

