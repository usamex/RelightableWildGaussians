#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return F.l1_loss(network_output, gt, reduction="mean")

def l2_loss(network_output, gt):
    return F.mse_loss(network_output, gt, reduction="mean")

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        window_size (int): Size of the window (number of elements).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: Normalized 1D Gaussian kernel of shape (window_size,).
    """
    center = window_size // 2
    values = [exp(-(x - center) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    kernel = torch.tensor(values, dtype=torch.float32)
    return kernel / kernel.sum()

def create_window(window_size: int, channel: int) -> torch.Tensor:
    """
    Creates a 2D Gaussian window.

    Args:
        window_size (int): Size of the window (number of elements).
        channel (int): Number of channels.

    Returns:
        torch.Tensor: Normalized 2D Gaussian window of shape (channel, 1, window_size, window_size).
    """
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.T
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (torch.Tensor): First image.
        img2 (torch.Tensor): Second image.
        window_size (int): Size of the window.
        size_average (bool): Whether to average the SSIM over the image.
    """
    channel = img1.size(-3) # Expects img1 to be (B, C, H, W)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.to(img1.device)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int, channel: int, size_average: bool) -> torch.Tensor:
    """
    Internal function to compute SSIM given a precomputed Gaussian window.

    Args:
        img1 (torch.Tensor): First image.
        img2 (torch.Tensor): Second image.
        window (torch.Tensor): 2D Gaussian window.
        window_size (int): Size of the window.
    """

    padding = window_size // 2
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-3)