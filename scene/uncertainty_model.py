import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import urllib.request
import io
import os
import numpy as np
from typing import Optional, Any
from torch import Tensor
from arguments import OptimizationParams
from utils.loss_utils import ssim


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    """Convert image to specified dtype with proper scaling."""
    if image.dtype == dtype:
        return image
    
    if image.dtype == np.uint8 and dtype == np.float32:
        return image.astype(dtype) / 255.0
    elif image.dtype == np.float32 and dtype == np.uint8:
        return (image * 255.0).astype(dtype)
    else:
        return image.astype(dtype)


# DINOv2 model loading function (simplified version)
def load_dinov2_model(backbone_name: str, pretrained: bool = True):
    """Load DINOv2 model - simplified version for this codebase."""
    try:
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', backbone_name, pretrained=pretrained)
        return model
    except Exception as e:
        print(f"Failed to load DINOv2 model {backbone_name}: {e}")
        # Fallback to a simple CNN backbone
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_size = 14
                self.embed_dim = 384
                self.num_heads = 12
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 384, 3, stride=2, padding=1),
                    nn.ReLU(),
                )
                
            def get_intermediate_layers(self, x, n=None, reshape=True):
                features = self.features(x)
                if reshape:
                    # Reshape to match DINOv2 output format
                    B, C, H, W = features.shape
                    features = features.view(B, C, H * W).transpose(1, 2)
                    features = features.view(B, H, W, C).permute(0, 3, 1, 2)
                return [features]
        
        return SimpleCNN()


def ssim_down(x, y, max_size=None):
    osize = x.shape[2:]
    if max_size is not None:
        scale_factor = max(max_size/x.shape[-2], max_size/x.shape[-1])
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')
    out = ssim(x, y, size_average=False).unsqueeze(1)
    if max_size is not None:
        out = F.interpolate(out, size=osize, mode='bilinear', align_corners=False)
    return out.squeeze(1)


def _ssim_parts(img1, img2, window_size=11):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma1 = torch.sqrt(sigma1_sq.clamp_min(0))
    sigma2 = torch.sqrt(sigma2_sq.clamp_min(0))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return luminance, contrast, structure


def msssim(x, y, max_size=None, min_size=200):
    raw_orig_size = x.shape[-2:]
    if max_size is not None:
        scale_factor = min(1, max(max_size/x.shape[-2], max_size/x.shape[-1]))
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')

    ssim_maps = list(_ssim_parts(x, y))
    orig_size = x.shape[-2:]
    while x.shape[-2] > min_size and x.shape[-1] > min_size:
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        ssim_maps.extend(tuple(F.interpolate(x, size=orig_size, mode='bilinear') for x in _ssim_parts(x, y)[1:]))
    out = torch.stack(ssim_maps, -1).prod(-1)
    if max_size is not None:
        out = F.interpolate(out, size=raw_orig_size, mode='bilinear')
    return out.mean(1)


def dino_downsample(x, max_size=None):
    if max_size is None:
        return x
    h, w = x.shape[2:]
    if max_size < h or max_size < w:
        scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
        nh = int(h * scale_factor)
        nw = int(w * scale_factor)
        nh = ((nh + 13) // 14) * 14
        nw = ((nw + 13) // 14) * 14
        x = F.interpolate(x, size=(nh, nw), mode='bilinear')
    return x


def assert_not_none(value: Optional[Any]) -> Any:
    assert value is not None
    return value


class UncertaintyModel(nn.Module):
    """Uncertainty prediction model based on DINOv2 backbone."""
    
    img_norm_mean: Tensor
    img_norm_std: Tensor

    def __init__(self, config: OptimizationParams):
        super().__init__()
        self.config = config
        # Load DINOv2 backbone
        self.backbone = load_dinov2_model(config.uncertainty_backbone, pretrained=True)
        self.patch_size = getattr(self.backbone, 'patch_size', 14)
        in_features = getattr(self.backbone, 'embed_dim', 384)
        
        # Segmentation head
        self.conv_seg = nn.Conv2d(in_features, 1, kernel_size=1)
        self.bn = nn.SyncBatchNorm(in_features)
        nn.init.normal_(self.conv_seg.weight.data, 0, 0.01)
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)

        # Image normalization parameters (DINOv2 standard)
        img_norm_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        img_norm_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("img_norm_mean", img_norm_mean / 255.)
        self.register_buffer("img_norm_std", img_norm_std / 255.)

        self._images_cache = {}

        # Freeze DINOv2 backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _get_pad(self, size):
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def _initialize_head_from_checkpoint(self):
        """Initialize segmentation head from pre-trained ADE20K checkpoint."""
        try:
            # ADA20 classes to ignore (these classes are typically problematic)
            cls_to_ignore = [13, 21, 81, 84]
            
            # Download pre-trained checkpoint
            backbone = self.config.uncertainty_backbone
            url = f"https://dl.fbaipublicfiles.com/dinov2/{backbone}/{backbone}_ade20k_linear_head.pth"
            
            print(f"Downloading pre-trained head from: {url}")
            with urllib.request.urlopen(url) as f:
                checkpoint_data = f.read()
            
            checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cpu")
            old_weight = checkpoint["state_dict"]["decode_head.conv_seg.weight"]
            
            # Initialize new weight tensor
            new_weight = torch.empty(1, old_weight.shape[1], 1, 1)
            nn.init.normal_(new_weight, 0, 0.0001)
            new_weight[:, cls_to_ignore] = old_weight[:, cls_to_ignore] * 1000
            
            # Apply weights
            nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)
            self.conv_seg.weight.data.copy_(new_weight)

            # Load batch normalization parameters
            bn_state_dict = {k[len("decode_head.bn."):]: v for k, v in checkpoint["state_dict"].items() 
                           if k.startswith("decode_head.bn.")}
            self.bn.load_state_dict(bn_state_dict)
            
            print("Successfully initialized head from pre-trained checkpoint")
            
        except Exception as e:
            print(f"Failed to initialize head from checkpoint: {e}")
            print("Using random initialization instead")

    def _get_dino_cached(self, x, cache_entry=None):
        if cache_entry is None or (cache_entry, x.shape) not in self._images_cache:
            with torch.no_grad():
                x = self.backbone.get_intermediate_layers(x, n=[self.backbone.num_heads-1], reshape=True)[-1]
            if cache_entry is not None:
                self._images_cache[(cache_entry, x.shape)] = x.detach().cpu()
        else:
            x = self._images_cache[(cache_entry, x.shape)].to(x.device)
        return x

    def _compute_cosine_similarity(self, x, y, _x_cache=None, _y_cache=None, max_size=None):
        # Normalize data
        h, w = x.shape[2:]
        if max_size is not None and (max_size < h or max_size < w):
            assert max_size % 14 == 0, "max_size must be divisible by 14"
            scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
            nh = int(h * scale_factor)
            nw = int(w * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
            y = F.interpolate(y, size=(nh, nw), mode='bilinear')
        x = (x - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        y = (y - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        x = F.pad(x, pads)
        padded_shape = x.shape
        y = F.pad(y, pads)
        with torch.no_grad():
            x = self._get_dino_cached(x, _x_cache)
            y = self._get_dino_cached(y, _y_cache)

        cosine = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        cosine: Tensor = F.interpolate(cosine, size=padded_shape[2:], mode="bilinear", align_corners=False)
        
        # Remove padding
        cosine = cosine[:, :, pads[2]:h+pads[2], pads[0]:w+pads[0]]
        if max_size is not None and (max_size < h or max_size < w):
            cosine = F.interpolate(cosine, size=(h, w), mode='bilinear', align_corners=False)
        return cosine.squeeze(1)
    
    def _forward_uncertainty_features(self, inputs: Tensor, _cache_entry: Optional[str] = None) -> Tensor:
        """Forward pass through the uncertainty network."""
        # Normalize input data using DINOv2 normalization
        inputs = inputs.to(self.img_norm_mean.device)
        inputs = (inputs - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        h, w = inputs.shape[2:]
        
        # Apply padding to make dimensions compatible with patch size
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in inputs.shape[:1:-1]))
        inputs = F.pad(inputs, pads)

        # Extract features using DINOv2 backbone
        x = self._get_dino_cached(inputs, _cache_entry)

        # Apply dropout and batch normalization
        x = F.dropout2d(x, p=self.config.uncertainty_dropout, training=self.training)
        x = self.bn(x)
        
        # Apply segmentation head
        logits = self.conv_seg(x)
        # Add bias term for proper initialization
        logits = logits + math.log(math.exp(1) - 1)

        # Apply softplus activation and rescale to input size
        logits = F.softplus(logits)
        logits = F.interpolate(logits, size=inputs.shape[2:], mode="bilinear", align_corners=False)
        logits = logits.clamp(min=self.config.uncertainty_clip_min)

        # Remove padding
        logits = logits[:, :, pads[2]:h+pads[2], pads[0]:w+pads[0]]
        return logits

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.img_norm_mean.device

    def forward(self, image: Tensor, _cache_entry: Optional[str] = None) -> Tensor:
        """Forward pass of the uncertainty model."""
        return self._forward_uncertainty_features(image, _cache_entry=_cache_entry)

    def setup_data(self, train_dataset, test_dataset):
        """Setup training and test datasets for the model."""
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def _load_image(self, img: np.ndarray) -> Tensor:
        """Load and preprocess image for the model."""
        return torch.from_numpy(np.moveaxis(convert_image_dtype(img, np.float32), -1, 0)[None]).to(self.device)

    def _scale_input(self, x: Tensor, max_size: Optional[int] = 504) -> Tensor:
        """Scale input tensor to maximum size while preserving aspect ratio."""
        h, w = x.shape[2:]
        if max_size is not None:
            scale_factor = min(max_size/h, max_size/w)
            if scale_factor >= 1:
                return x
            nw = int(w * scale_factor)
            nh = int(h * scale_factor)
            # Ensure dimensions are compatible with patch size (14x14)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
        return x

    def _dino_plus_ssim(self, gt, prediction, _cache_entry=None, max_size=None):
        gt_down = dino_downsample(gt, max_size=max_size)
        prediction_down = dino_downsample(prediction, max_size=max_size)
        dino_cosine = self._compute_cosine_similarity(
            gt_down,
            prediction_down,
            _x_cache=_cache_entry).detach()
        dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
        msssim_part = 1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
        return torch.min(dino_part, msssim_part)

    def _compute_losses(self, gt, prediction, prefix='', _cache_entry=None):
        uncertainty = self(self._scale_input(gt, self.config.uncertainty_dino_max_size), _cache_entry=_cache_entry)
        log_uncertainty = torch.log(uncertainty)
        # _dssim_go = dssim_go(gt, prediction, size_average=False).unsqueeze(1).clamp_max(self.config.uncertainty_dssim_clip_max)
        # _dssim_go = 1 - ssim(gt, prediction).unsqueeze(1)
        _ssim = ssim_down(gt, prediction, max_size=400).unsqueeze(1)
        _msssim = msssim(gt, prediction, max_size=400, min_size=80).unsqueeze(1)

        if self.config.uncertainty_mode == "l2reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = 1 / (2 * uncertainty.pow(2))
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "l1reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = 1 / uncertainty
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "dino":
            # loss_mult = 1 / (2 * uncertainty.pow(2))
            # loss_mult = 1 / uncertainty
            # Compute dino loss
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=350)
            prediction_down = dino_downsample(prediction, max_size=350)
            dino_cosine = self._compute_cosine_similarity(
                gt_down,
                prediction_down,
                _x_cache=_cache_entry).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            uncertainty_loss = dino_part * dino_downsample(loss_mult, max_size=350)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(loss_mult, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = loss_mult.clamp_max(3)

        elif self.config.uncertainty_mode == "dino+mssim":
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=350)
            prediction_down = dino_downsample(prediction, max_size=350)
            dino_cosine = self._compute_cosine_similarity(
                gt_down,
                prediction_down,
                _x_cache=_cache_entry).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            msssim_part = 1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
            uncertainty_loss = torch.min(dino_part, msssim_part) * dino_downsample(loss_mult, max_size=350)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(loss_mult, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = loss_mult.clamp_max(3)

        else:
            raise ValueError(f"Invalid uncertainty_mode: {self.config.uncertainty_mode}")

        beta = log_uncertainty.mean()
        loss = uncertainty_loss.mean() + self.config.uncertainty_regularizer_weight * beta

        ssim_discounted = (_ssim * loss_mult).sum() / loss_mult.sum()
        mse = torch.pow(gt - prediction, 2)
        mse_discounted = (mse * loss_mult).sum() / loss_mult.sum()
        psnr_discounted = 10 * torch.log10(1 / mse_discounted)

        metrics = {
            f"{prefix}loss": loss.item(),
            f"{prefix}ssim": _ssim.mean().item(),
            f"{prefix}msssim": _msssim.mean().item(),
            f"{prefix}ssim_discounted": ssim_discounted.item(),
            f"{prefix}mse_discounted": mse_discounted.item(),
            f"{prefix}psnr_discounted": psnr_discounted.item(),
            f"{prefix}beta": beta.item(),
        }
        return loss, metrics, loss_mult.detach()

    def get_loss(self, gt_image: Tensor, image: Tensor, prefix: str = '', _cache_entry: Optional[str] = None):
        """Compute uncertainty loss and metrics."""
        gt_torch = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)
        loss, metrics, loss_mult = self._compute_losses(gt_torch, image, prefix, _cache_entry=_cache_entry)
        loss_mult = loss_mult.squeeze(0)
        metrics[f"{prefix}uncertainty_loss"] = metrics.pop(f"{prefix}loss")
        metrics.pop(f"{prefix}ssim")
        return loss, metrics, loss_mult

    @staticmethod
    def load(path: str, config: Optional[OptimizationParams] = None):
        """Load uncertainty model from checkpoint."""
        checkpoint_path = os.path.join(path, "uncertainty_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Uncertainty checkpoint not found at {checkpoint_path}")
            
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        if config is None:
            # Try to load config from checkpoint
            config_dict = ckpt.pop("config", {})
            config = OptimizationParams(None)  # Create default config
            # Update config with saved values
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        model = UncertaintyModel(config)
        model.load_state_dict(ckpt, strict=False)
        return model

    def save(self, path: str):
        """Save uncertainty model checkpoint."""
        os.makedirs(path, exist_ok=True)
        state = self.state_dict()
        # Save config as dict
        config_dict = {}
        for key in dir(self.config):
            if not key.startswith('_') and not callable(getattr(self.config, key)):
                config_dict[key] = getattr(self.config, key)
        state["config"] = config_dict
        torch.save(state, os.path.join(path, "uncertainty_checkpoint.pth"))
