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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from scene.NVDIFFREC.light import EnvironmentLight
from utils.normal_utils import compute_normal_world_space

# Technically DONE, probably it will fail while running but all good.
# Multiplier is a tensor that indicates which gaussians are flipped and which are not. (Check lumigauss)

def get_shaded_colors(envlight: EnvironmentLight, pos: torch.tensor, view_pos: torch.tensor, normal: torch.tensor=None, albedo: torch.tensor=None, 
                       roughness:torch.tensor=None, metalness:torch.tensor=None, specular:bool=True):
    if metalness is not None:
        metalness = metalness[None, None, ...]
    colors_precomp, brdf_pkg = envlight.shade(gb_pos=pos[None, None, ...], gb_normal=normal[None, None, ...], albedo=albedo[None, None, ...],
                            view_pos=view_pos[None, None, ...], kr=roughness[None, None, ...], km=metalness, specular=specular)
    return colors_precomp, brdf_pkg


def render(viewpoint_camera, pc : GaussianModel, envlight: EnvironmentLight, sky_sh: torch.tensor, sky_sh_degree: int, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, debug=True, specular=True, fix_sky=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=-1, # sh_degree is -1 for envlight (idk why) to not use SHs at all!!
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    sky_mask = viewpoint_camera.sky_mask.cuda().squeeze()
    sky_gaussians_mask = pc.get_is_sky.squeeze() # (N)
    positions = pc.get_xyz # (N, 3)
    albedo = pc.get_albedo # (N, 3)
    roughness = pc.get_roughness # (N, 1)
    metalness = pc.get_metalness # (N,1)
    view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3)
    dir_pp = (pc.get_xyz - view_pos) # Vector from camera center to each gaussian (N, 3)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # Unit vectors pointing from camera to each gaussian (N, 3)

    normal, multiplier = compute_normal_world_space(rotations, scales, viewpoint_camera.world_view_transform, positions) # TODO add the multiplier to the codebase!!
    colors_precomp, diffuse_color, specular_color, sky_color = (torch.zeros(positions.shape[0], 3, dtype=torch.float32, device="cuda") for _ in range(4))

    # Compute color for the foreground Gaussians
    color_fg_gaussians, brdf_pkg = get_shaded_colors(envlight=envlight, pos=positions[~sky_gaussians_mask],
                                                          view_pos=view_pos[~sky_gaussians_mask], normal=normal[~sky_gaussians_mask],
                                                          albedo=albedo,
                                                          roughness=roughness, metalness=metalness,
                                                          specular=specular)
    colors_precomp[~sky_gaussians_mask] = color_fg_gaussians.squeeze() # Give color to foreground Gaussians

    # Compute color for the sky (background) Gaussians
    if fix_sky:
        colors_precomp[sky_gaussians_mask] = torch.ones_like(positions[sky_gaussians_mask])
    else:
        sky_sh2rgb = eval_sh(sky_sh_degree, sky_sh.transpose(1,2), dir_pp_normalized[sky_gaussians_mask])
        color_sky_gaussians = torch.clamp_min(sky_sh2rgb + 0.5, 0.0)
        colors_precomp[sky_gaussians_mask] = color_sky_gaussians
                                           
    # Set diffuse and specular colors for the foreground Gaussians
    diffuse_color[sky_gaussians_mask] = torch.zeros_like(colors_precomp[sky_gaussians_mask])
    diffuse_color[~sky_gaussians_mask] = brdf_pkg['diffuse'].squeeze()
    specular_color[sky_gaussians_mask] = torch.zeros_like(colors_precomp[sky_gaussians_mask])
    specular_color[~sky_gaussians_mask] = brdf_pkg['specular'].squeeze()

    # Set sky color for the sky (background) Gaussians
    sky_color[sky_gaussians_mask] = colors_precomp[sky_gaussians_mask]
    sky_color[~sky_gaussians_mask] = torch.zeros_like(colors_precomp[~sky_gaussians_mask])

    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets