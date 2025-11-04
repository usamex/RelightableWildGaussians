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
from utils.sh_utils import eval_sh
from utils.general_utils import rand_hemisphere_dir
import random


TINY_NUMBER = 1e-6


def l1_loss(network_output, gt, mask=None):
    if torch.is_tensor(mask):
        return ((torch.abs((network_output*mask - gt*mask))).sum()) / (network_output.shape[0]*mask.sum() + TINY_NUMBER)
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt, mask=None):
    if torch.is_tensor(mask):
        return (((network_output*mask - gt*mask) ** 2).sum()) / (network_output.shape[0]*mask.sum() + TINY_NUMBER )
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if torch.is_tensor(mask):
            return (ssim_map*mask).sum() / (ssim_map.shape[0]*mask.sum() + 1e-6 )
        return ssim_map.mean()
    else:
        return ssim_map.mean(-3)

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
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
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

def depth_loss_gaussians(mean_depth_sky, mean_depth_non_sky, gamma = 0.02):
    """The function compares the difference of the average depth of sky and non sky gaussians using an exponential function."""
    loss = torch.exp(-gamma*(mean_depth_sky-mean_depth_non_sky))
    return loss

def envlight_loss(envlight_sh: torch.tensor, sh_degree: int, normals: torch.Tensor, N_dirs: int = 1000, normals_subset_size = 100):
    """
    Regularization on environment lighting coefficients: incoming light should belong to R+.
    The loss is computed on a random subset of the input normals. For each normal N random directions
    in the hemisphere around it are sampled, then the irradiance corresponding to such direction is computed
    and the minimum between its values and 0 is taken.
    Args:
        envlight: environment lighting SH coefficients,
        normals: normal vectors of shape [..., 3],
        N_dirs: number of viewing directions samples,
        normals_subset_size: number of normal samples
    """
    assert normals.shape[-1] == 3 , "error: normals must have size  [...,3]" # TODO: This might be wrong

    if normals.shape[0] > normals_subset_size:
        normals_rand_subset = random.sample(range(0, normals_subset_size), normals_subset_size)
        normals = normals[normals_rand_subset]

    # generate N_dirs random viewing directions in the hemisphere centered in n for each n in normals
    rand_hemisphere_dirs = rand_hemisphere_dir(N_dirs, normals) # (..., N, 3)
    # evaluate SH coefficients of env light
    light = eval_sh(sh_degree, envlight_sh.transpose(0,1), rand_hemisphere_dirs)
    # extract negative values
    light = torch.minimum(light, torch.zeros_like(light))
    # average negative light values over number of viewing direction samples
    avg_light_per_normal = torch.mean(light, dim = 1)
    # average over normals
    avg_light = torch.mean(avg_light_per_normal, dim = 0)
    # take squared 2 norm
    envlight_loss = torch.mean((avg_light)**2)

    return envlight_loss


def envl_sh_loss(sh_env: torch.tensor, sh_degree, N_samples: int=10):

    shs_view = sh_env.repeat(N_samples, 1, 1)
    view_dir_unnorm = torch.empty(shs_view.shape[0], 3, device=shs_view.device).uniform_(-1,1)
    view_dir = view_dir_unnorm / view_dir_unnorm.norm(dim=1, keepdim=True)
    envl_val = eval_sh(sh_degree, shs_view.transpose(1,2), view_dir)

    #Constrain illumination values to R+
    sh_envl_loss = penalize_outside_range(envl_val.view(-1), 0.0,torch.inf)
    
    return sh_envl_loss



def penalize_outside_range(tensor, lower_bound=0.0, upper_bound=1.0):
    error = 0
    below_lower_bound = tensor[tensor < lower_bound]
    above_upper_bound = tensor[tensor > upper_bound]
    if below_lower_bound.numel():
        error += torch.mean((below_lower_bound - lower_bound) ** 2)
    if above_upper_bound.numel():
        error += torch.mean((above_upper_bound - upper_bound) ** 2)
    return error


def min_scale_loss(radii, gaussians):
    visibility_filter = radii > 0
    try:
        if visibility_filter.sum() > 0: # consider just visible fg gaussians
            scale = gaussians.get_scaling[visibility_filter & ~gaussians.get_is_sky.squeeze()]
            scale = scale
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0] # take minimum scales
            return min_scale_loss.mean()
    except Exception as e:
        raise RuntimeError(f"Failed to compute min_scale_loss: {e}")

def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        if mask.shape[0] == x.shape[0]:
            return torch.sum((x - y) * (x - y) * mask.unsqueeze(0)) / (torch.sum(mask) + TINY_NUMBER)
        else:
            return torch.sum((x - y) * (x - y) * mask.unsqueeze(0)) / (torch.sum(mask)*x.shape[0] + TINY_NUMBER)

def img2mae(x, y, mask=None):
    if mask is None:
        return torch.mean(torch.abs(x - y))
    else:
        if mask.shape[0] == x.shape[0]:
            return torch.sum(torch.abs(x - y) * mask.unsqueeze(0)) / (torch.sum(mask) + TINY_NUMBER)
        else:
            return torch.sum(torch.abs(x - y) * mask.unsqueeze(0)) / (torch.sum(mask) * x.shape[0] + TINY_NUMBER)

def mse2psnr(x):
    return -10. * torch.log(torch.tensor(x)+TINY_NUMBER) / torch.log(torch.tensor(10))

