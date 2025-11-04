# Relightable Wild Gaussians

A PyTorch implementation of 3D Gaussian Splatting with relighting capabilities for outdoor scenes. This project extends the original [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) to support environment lighting reconstruction, uncertainty modeling, and physics-based relighting.

## Features

- **Relightable 3D Reconstruction**: Decompose scenes into geometry, albedo, and environment lighting
- **Environment Lighting Modeling**: Spherical harmonics-based environment light representation
- **Uncertainty-Aware Training**: DINO-based uncertainty estimation for robust optimization
- **Sky/Outdoor Scene Support**: Special handling for sky regions and outdoor illumination
- **Physics-Based Rendering**: BRDF-based rendering with diffuse components

## Installation

### Prerequisites

- CUDA-capable GPU (CUDA 11.0 or higher recommended)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [COLMAP](https://colmap.github.io/) (for data preprocessing)

### Environment Setup

1. Clone the repository with submodules:
```bash
git clone https://github.com/yourusername/RelightableWildGaussians.git --recursive
cd RelightableWildGaussians
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate rwg
```

3. Install nvdiffrast (if not already included):
```bash
cd nvdiffrast
pip install .
cd ..
```

## Data Preparation

### Using COLMAP for Custom Data

Convert your images to COLMAP format:

```bash
python convert.py -s <path_to_your_images> --resize
```

This script will:
- Run COLMAP feature extraction and matching
- Perform bundle adjustment
- Undistort images
- Create resized versions (2x, 4x, 8x downscaling)

### Expected Data Structure

```
<dataset_path>/
├── images/          # Original images
├── sparse/          # COLMAP sparse reconstruction
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── sky_masks/       # (Optional) Sky segmentation masks
└── occluders_masks/ # (Optional) Transient occluder masks
```

## Training

### Basic Training

```bash
python train.py -s <path_to_dataset> -m <output_path>
```

### Training with Custom Parameters

```bash
python train.py \
  -s <path_to_dataset> \
  -m <output_path> \
  --iterations 200000 \
  --envlight_sh_degree 4 \
  --sky_sh_degree 1 \
  --lambda_sky_brdf 0.5 \
  --with_mlp
```

### Key Training Arguments

**Model Parameters:**
- `--source_path, -s`: Path to dataset
- `--model_path, -m`: Output directory for trained model
- `--images`: Subdirectory containing images (default: `images`)
- `--resolution, -r`: Image resolution for training (-1 for original)
- `--white_background`: Use white background instead of black
- `--envlight_sh_degree`: Degree of spherical harmonics for environment lighting (default: 4)
- `--sky_sh_degree`: Degree of SH for sky (default: 1)
- `--with_mlp`: Use MLP for appearance embedding (recommended)

**Optimization Parameters:**
- `--iterations`: Number of training iterations (default: 200000)
- `--position_lr_init`: Initial learning rate for positions (default: 0.000016)
- `--lambda_dssim`: Weight for SSIM loss (default: 0.2)
- `--lambda_normal`: Weight for normal regularization (default: 0.05)
- `--lambda_dist`: Weight for distortion loss (default: 0.0)
- `--lambda_sky_brdf`: Weight for sky BRDF loss (default: 0.5)
- `--densify_grad_threshold`: Threshold for densification (default: 0.0002)

**Uncertainty Parameters:**
- `--uncertainty_mode`: Uncertainty model type (default: "dino")
- `--uncertainty_backbone`: DINO backbone (default: "dinov2_vits14_reg")
- `--uncertainty_warmup_start`: Iteration to start uncertainty warmup (default: 35000)
- `--uncertainty_warmup_iters`: Duration of warmup (default: 5000)

### Training Output

Training creates the following structure:
```
<output_path>/
├── point_cloud/
│   └── iteration_<N>/
│       └── point_cloud.ply
├── chkpnt<N>.pth          # Model checkpoints
├── appearance_lut.json     # Appearance embedding lookup table
└── cfg_args                # Training configuration
```

## Rendering

### Render Test Views

```bash
python render.py -m <path_to_trained_model> --iteration <iteration_number>
```

## Evaluation

### Compute Metrics

Evaluate rendering quality using PSNR, SSIM, and LPIPS:

```bash
python metrics.py -m <path_to_model_outputs>
```

The script will generate:
- `results.json`: Average metrics per method
- `per_view.json`: Per-image metrics

## Project Structure

```
RelightableWildGaussians/
├── train.py                    # Main training script
├── render.py                   # Rendering script
├── view.py                     # Interactive viewer
├── convert.py                  # COLMAP data conversion
├── metrics.py                  # Evaluation metrics
├── arguments/                  # Argument parsers
│   └── __init__.py
├── scene/                      # Scene and model definitions
│   ├── gaussian_model.py       # 3D Gaussian model
│   ├── cameras.py              # Camera definitions
│   ├── dataset_readers.py      # Dataset loaders
│   ├── uncertainty_model.py    # Uncertainty estimation
│   └── NVDIFFREC/              # Lighting and rendering utilities
├── gaussian_renderer/          # Rendering implementation
├── utils/                      # Utility functions
│   ├── loss_utils.py           # Loss functions
│   ├── sh_utils.py             # Spherical harmonics utilities
│   ├── image_utils.py          # Image processing
│   └── ...
├── scripts/                    # Evaluation scripts
└── submodules/                 # External dependencies
    ├── diff-surfel-rasterization/
    └── simple-knn/
```

## Technical Details

### Lighting Model

The environment lighting is represented using spherical harmonics (SH) up to degree 4 by default. The sky is modeled separately using lower-degree SH (degree 1). Each training view has its own environment lighting parameters learned through an MLP or direct optimization.

### Uncertainty Modeling

The uncertainty model uses DINO features (DINOv2) to identify regions with unreliable reconstructions (e.g., transient objects, reflections). During training, the loss is weighted by uncertainty to focus on reliable regions.

### BRDF Decomposition

The model decomposes each Gaussian's color into:
- **Albedo**: Intrinsic surface color
- **Diffuse shading**: Lambertian lighting response

## Advanced Usage

### Resume Training from Checkpoint

```bash
python train.py -s <dataset> -m <model_path> --start_checkpoint <path_to_checkpoint>
```

### Custom Test Iterations

```bash
python train.py -s <dataset> -m <model_path> \
  --test_iterations 5000 10000 20000 50000 \
  --save_iterations 5000 10000 20000 50000
```

## Citation

If you use this code in your research, please cite the original 3D Gaussian Splatting paper:

```bibtex
@Article{kerbl3Dgaussians,
    author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal = {ACM Transactions on Graphics},
    year    = {2023},
    volume  = {42},
    number  = {4},
    pages   = {Article 139},
}
```

## License

This software is licensed under the Gaussian-Splatting License. See [LICENSE.md](LICENSE.md) for details.

**For non-commercial research and evaluation use only.** For commercial use, please contact:
- Inria: stip-sophia.transfert@inria.fr
- MPII: mpii-tech-transfer@mpi-inf.mpg.de

## Acknowledgments

This project builds upon:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Inria and MPII
- [NVDIFFREC](https://github.com/NVlabs/nvdiffrec) for differentiable rendering utilities
- [DINOv2](https://github.com/facebookresearch/dinov2) for uncertainty estimation
- [diff-surfel-rasterization 2DGS](https://github.com/hbb1/diff-surfel-rasterization) for efficient rasterization

---

**Note**: This is an active research project. Features and APIs may change.
