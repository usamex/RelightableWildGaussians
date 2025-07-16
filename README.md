# RelightableWildGaussians Dataset Reader

This module provides functionality to read and process datasets in the Lumigauss format for the RelightableWildGaussians project.

## Dataset Structure

The dataset should be structured as follows:

```
dataset_root/
├── images/            # RGB images
├── masks/             # Segmentation masks (optional)
├── sparse/            # COLMAP reconstruction
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── <split_file>.csv   # Train/test split information (optional)
```

The split file (if provided) should be a CSV file with a header row and columns for the image filename and split category:

```
filename;split
image1.jpg;train
image2.jpg;test
...
```

## Basic Usage

```python
from RelightableWildGaussians.data_reader import LumigaussDataset

# Initialize dataset
dataset = LumigaussDataset(
    path="/path/to/dataset",
    eval_split=True,               # Whether to split into train/test
    split_file="trevi_split.csv",  # CSV file with train/test split info
    images_dir="images",           # Directory containing images
    masks_dir="masks"              # Directory containing masks
)

# Load dataset
dataset.load()

# Access data
train_cameras = dataset.get_train_cameras()
test_cameras = dataset.get_test_cameras()
point_cloud = dataset.get_point_cloud()
normalization = dataset.get_normalization()

# Access camera information
for camera in train_cameras:
    print(f"Image: {camera.image_name}")
    print(f"Resolution: {camera.width}x{camera.height}")
    # Access other camera properties:
    # - camera.R: Rotation matrix
    # - camera.T: Translation vector
    # - camera.image: PIL Image
    # - camera.mask: PIL mask image
    # - camera.FovX, camera.FovY: Field of view
```

## Requirements

- NumPy
- PIL (Pillow)
- pandas
- plyfile 