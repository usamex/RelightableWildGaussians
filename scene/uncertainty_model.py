from scene import dinov2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import io
import urllib.request
from typing import Optional
from torch import Tensor
import itertools
from utils.loss_utils import ssim_down, msssim, dino_downsample
from utils.image_utils import convert_image_dtype
from utils.general_utils import assert_not_none

class UncertaintyModel(nn.Module):
    img_norm_mean: Tensor
    img_norm_std: Tensor

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = getattr(dinov2, config.uncertainty_backbone)(pretrained=True)
        self.patch_size = self.backbone.patch_size
        in_features = self.backbone.embed_dim
        self.conv_seg = nn.Conv2d(in_features, 1, kernel_size=1)
        self.bn = nn.SyncBatchNorm(in_features)
        nn.init.normal_(self.conv_seg.weight.data, 0, 0.01)
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)

        img_norm_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        img_norm_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("img_norm_mean", img_norm_mean / 255.)
        self.register_buffer("img_norm_std", img_norm_std / 255.)

        self._images_cache = {}

        # Freeze dinov2 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _get_pad(self, size):
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def _initialize_head_from_checkpoint(self):
        # ADA20 classes to ignore
        cls_to_ignore = [13, 21, 81, 84]
        # Pull the checkpoint
        backbone = self.config.uncertainty_backbone
        url = f"https://dl.fbaipublicfiles.com/dinov2/{backbone}/{backbone}_ade20k_linear_head.pth"
        with urllib.request.urlopen(url) as f:
            checkpoint_data = f.read()
        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cpu")
        old_weight = checkpoint["state_dict"]["decode_head.conv_seg.weight"]
        new_weight = torch.empty(1, old_weight.shape[1], 1, 1)
        nn.init.normal_(new_weight, 0, 0.0001)
        new_weight[:, cls_to_ignore] = old_weight[:, cls_to_ignore] * 1000
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)
        self.conv_seg.weight.data.copy_(new_weight)

        # Load the bn data
        self.bn.load_state_dict({k[len("decode_head.bn."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("decode_head.bn.")})

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
    
    def _forward_uncertainty_features(self, inputs: Tensor, _cache_entry=None) -> Tensor:
        # Normalize data
        inputs = (inputs - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        h, w = inputs.shape[2:]
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in inputs.shape[:1:-1]))
        inputs = F.pad(inputs, pads)

        x = self._get_dino_cached(inputs, _cache_entry)

        x = F.dropout2d(x, p=self.config.uncertainty_dropout, training=self.training)
        x = self.bn(x)
        logits = self.conv_seg(x)
        # We could also do this using weight init, 
        # but we want to have a prior then doing L2 regularization
        logits = logits + math.log(math.exp(1) - 1)

        # Rescale to input size
        logits = F.softplus(logits)
        logits: Tensor = F.interpolate(logits, size=inputs.shape[2:], mode="bilinear", align_corners=False)
        logits = logits.clamp(min=self.config.uncertainty_clip_min)

        # Add padding
        logits = logits[:, :, pads[2]:h+pads[2], pads[0]:w+pads[0]]
        return logits

    @property
    def device(self):
        return self.img_norm_mean.device

    def forward(self, image: Tensor, _cache_entry=None):
        return self._forward_uncertainty_features(image, _cache_entry=_cache_entry)

    def setup_data(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def _load_image(self, img):
        return torch.from_numpy(np.moveaxis(convert_image_dtype(img, np.float32), -1, 0)[None]).to(self.device)

    def _scale_input(self, x, max_size: Optional[int] = 504):
        h, w = nh, nw = x.shape[2:]
        if max_size is not None:
            scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
            if scale_factor >= 1:
                return x
            nw = int(w * scale_factor)
            nh = int(h * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
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

    def get_loss(self, gt_image, image, prefix='', _cache_entry=None):
        gt_torch = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)
        loss, metrics, loss_mult = self._compute_losses(gt_torch, image, prefix, _cache_entry=_cache_entry)
        loss_mult = loss_mult.squeeze(0)
        metrics[f"{prefix}uncertainty_loss"] = metrics.pop(f"{prefix}loss")
        metrics.pop(f"{prefix}ssim")
        return loss, metrics, loss_mult

    @staticmethod
    def load(path: str):
        ckpt = torch.load(os.path.join(path, "checkpoint.pth"), map_location="cpu")
        config = ckpt.pop("config") # TODO: Fix this
        model = UncertaintyModel(config)
        model.load_state_dict(ckpt, strict=False)
        return model

    def save(self, path: str):
        state = self.state_dict()
        state["config"] = self.config # TODO: Fix this
        torch.save(state, os.path.join(path, "checkpoint.pth"))

