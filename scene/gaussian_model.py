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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, cartesian_to_polar_coord
from utils.camera_utils import get_scene_center

from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_scaling_rotation, sample_points_on_unit_hemisphere
from scene.uncertainty_model import UncertaintyModel
from arguments import UncertaintyParams
from arguments import ModelParams
from scene.net_models import MLPNet
from scene.NVDIFFREC import EnvironmentLight

class GaussianModel:

    def setup_functions(self, model_config : ModelParams, length_train_cameras : int):

        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation): # This is because of 2D Gaussians
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if self.with_mlp:
            self.setup_mlp(model_config, length_train_cameras)

    def __init__(self, uncertainty_config : UncertaintyParams = None, model_config : ModelParams = None, length_train_cameras : int = 0):
        self._xyz = torch.empty(0, device="cuda")
        self._albedo = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self._is_sky = torch.empty(0, device="cuda")
        self._sky_radius = torch.empty(0, device="cuda")
        self._sky_gauss_center = torch.empty(0, device="cuda")
        self._sky_angles = torch.empty(0, device="cuda")

        self.material_properties_activation = torch.sigmoid
        self.default_albedo = 1.0

        self.denom = torch.empty(0, device="cuda")
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.with_mlp = model_config.with_mlp
        if uncertainty_config is not None:
            self.uncertainty_model = UncertaintyModel(uncertainty_config).to("cuda")
        else:
            self.uncertainty_model = None

        self.envlight = EnvironmentLight(base = torch.empty(((model_config.envlight_sh_degree +1)**2), 3),
                                            sh_degree=model_config.envlight_sh_degree)
        self.setup_functions(model_config, length_train_cameras)

    def setup_mlp(self, model_config : ModelParams, length_train_cameras : int):
        print("Setting up MLP")
        self.mlp: MLPNet = MLPNet(sh_degree_envlight=model_config.envlight_sh_degree, sh_degree_sky=model_config.sky_sh_degree, embedding_dim=model_config.embeddings_dim).cuda()
        self.embedding = torch.nn.Embedding(length_train_cameras, model_config.embeddings_dim).cuda()

    def capture(self):
        return (
            self._xyz,
            self._albedo,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self._is_sky,
            self._sky_radius,
            self._sky_gauss_center,
            self._sky_angles,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self._xyz, 
        self._albedo,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        self._is_sky,
        self._sky_radius,
        self._sky_gauss_center,
        self._sky_angles,
        opt_dict, 
        self.spatial_lr_scale) = model_args


        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        if torch.sum(self._is_sky) == 0:
            return self._xyz
        else:
            sky_xyz = self.get_sky_xyz
            all_xyz = torch.empty((sky_xyz.shape[0] + self._xyz.shape[0], 3), dtype=self._xyz.dtype, device=self._xyz.device)
            all_xyz[~self._is_sky.squeeze()] = self._xyz
            all_xyz[self._is_sky.squeeze()] = sky_xyz
            return all_xyz

    @property
    def get_sky_xyz(self): # TODO: I'm not sure if we want to get the angles like this
        sky_angles = self.get_sky_angles
        # In COLMAP coordinate system
        x = torch.sin(sky_angles[...,0]) * torch.sin(sky_angles[...,1])
        y = -torch.cos(sky_angles[...,0])
        z = torch.sin(sky_angles[...,0]) * torch.cos(sky_angles[...,1])
        sky_xyz = self._sky_radius * torch.stack([x, y, z], dim=-1) +  self._sky_gauss_center.squeeze()
        return sky_xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_albedo(self):
        return self.material_properties_activation(self._albedo)

    @property
    def get_sky_angles(self): # TODO: Not sure if this is correct
        # theta admitted range: [0, pi/2], phi admitted range: [-pi/2, pi/2]
        theta_mask = (self._sky_angles[...,0] < 0) | (self._sky_angles[...,0] > torch.pi/2)
        phi_mask = (self._sky_angles[...,1] < -torch.pi/2) | (self._sky_angles[...,1] > torch.pi/2)

        sky_theta = torch.where(theta_mask, torch.clamp(self._sky_angles[...,0], 0, torch.pi/2),
                                self._sky_angles[...,0])
        sky_phi = torch.where(phi_mask, torch.clamp(self._sky_angles[...,1], -torch.pi/2, torch.pi/2),
                              self._sky_angles[...,1])

        return torch.cat((sky_theta.unsqueeze(1), sky_phi.unsqueeze(1)), dim=1)

    @property
    def get_is_sky(self):
        return self._is_sky

    def compute_embedding(self, emb_idx):
        return self.embedding(torch.full((1,),emb_idx).cuda())

    def compute_env_sh(self, emb_idx):
        if self.with_mlp:
            return self.mlp(self.compute_embedding(emb_idx))
        else:
            raise NotImplementedError("Computing Environment Lighting without MLP is not implemented")

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # TODO: Why not use this???
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda") # TODO: Should we try 0 as well?

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._albedo = nn.Parameter(self.default_albedo * torch.ones((fused_point_cloud.shape[0], 3), device="cuda").requires_grad_(True))
        self._is_sky =  torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.bool, device="cuda")

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    @torch.no_grad()
    def get_sky_xyz_init(self, cameras): # TODO: I'm not sure if this is also applicable for 2DGS
        """Adapted from https://arxiv.org/abs/2407.08447"""
        mean = self._xyz.mean(0)[None]
        sky_distance = torch.quantile(torch.linalg.norm(self._xyz - mean, 2, -1), 0.99)
        scene_center = torch.tensor(get_scene_center(cameras), dtype=torch.float32, device="cuda").T
        num_sky_points = int(5000 * sky_distance.item())
        points = sample_points_on_unit_hemisphere(num_sky_points)
        points = points.to("cuda")
        points = points * sky_distance
        points = points + scene_center
        gmask = torch.zeros((points.shape[0],), dtype=torch.bool, device=points.device)
        for cam in cameras:
            uv = cam.camera_project(points[torch.logical_not(gmask)])
            mask = torch.logical_not(torch.isnan(uv).any(-1))
            # Only top 2/3 of the image
            assert cam.image_width is not None and cam.image_height is not None
            mask = torch.logical_and(mask, uv[..., -1] < 2/3 * cam.image_height)
            gmask[torch.logical_not(gmask)] = torch.logical_or(gmask[torch.logical_not(gmask)], mask)

        return points[gmask], sky_distance, scene_center

    @torch.no_grad()
    def augment_with_sky_gaussians(self, cameras): # TODO: I'm not sure if this is also applicable for 2DGS
        sky_xyz, sky_distance, sky_gauss_center = self.get_sky_xyz_init(cameras)
        self._sky_gauss_center = sky_gauss_center
        print(f"Adding {sky_xyz.shape[0]} sky Gaussians")
        # Initialize polar coordinates:
        self._sky_radius = nn.Parameter(torch.tensor(sky_distance, dtype=torch.float32, device="cuda", requires_grad=True))
        sky_angles = cartesian_to_polar_coord(sky_xyz, self._sky_gauss_center.squeeze(), self._sky_radius)
        self._sky_angles = nn.Parameter(sky_angles).requires_grad_(True)

        sky_opacity =  inverse_sigmoid(0.1 * torch.ones((sky_xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        self._opacity = nn.Parameter(torch.cat([self._opacity, sky_opacity]))

        self._is_sky = torch.cat((self._is_sky, torch.ones((sky_xyz.shape[0], 1), dtype=torch.bool, device=self._is_sky.device)), dim=0)


        dist2 = torch.clamp_min(distCUDA2(sky_xyz.float().cuda()), 0.0000001)
        sky_scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, self._scaling.shape[1]) # Since it is 2DGS it will be 2
        sky_rots = torch.zeros((sky_xyz.shape[0], 4), device=self._rotation.device)
        sky_rots[:, 0] = 1
        self._rotation = nn.Parameter(torch.cat([self._rotation, sky_rots.requires_grad_(True)], dim=0))
        self._scaling = nn.Parameter(torch.cat([self._scaling, sky_scales.requires_grad_(True)]))
        sky_max_radii2D = torch.zeros((sky_xyz.shape[0]), device=self.max_radii2D.device)
        self.max_radii2D = nn.Parameter(torch.cat([self.max_radii2D, sky_max_radii2D]))

    def training_setup(self, training_args: dict):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._sky_radius], 'lr': training_args.sky_radius_lr, "name": "sky_radius"},
            {'params': [self._sky_angles], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "sky_angles"},

        ]

        if self.uncertainty_model is not None:
            l.append({'params': list(self.uncertainty_model.parameters()), 'lr': training_args.uncertainty_lr, "name": "uncertainty_model"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15) # TODO: Take a look at this after
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        l_env = [
            {'params': [*self.embedding.parameters()], 'lr': training_args.embedding_lr, "name": "embeddings"},
            {'params': [*self.mlp.parameters()], 'lr': training_args.mlp_lr, "name": "mlp"},
        ]
        self.optimizer_env = torch.optim.Adam(l_env, lr=0.0, eps=1e-15) # TODO: Try it with only one optimizer
        print('Env optimizer has parameters: ', [p["name"] for p in self.optimizer_env.param_groups])


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz" or param_group["name"] == "sky_angles":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('is_sky')
        if torch.sum(self._is_sky) > 0:
            l.append('sky_radius')
            for i in range(self._sky_gauss_center.shape[1]):
                l.append('sky_gauss_center_{}'.format(i))
            for i in range(self._sky_angles.shape[1]):
                l.append('sky_angles_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        sky_pts_mask = self.get_is_sky.squeeze()
        fg_pts_mask = ~self.get_is_sky.squeeze()
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        albedo = torch.ones_like(self.get_xyz)
        albedo[fg_pts_mask] = self._albedo
        albedo = albedo.detach().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        is_sky = self._is_sky.cpu().numpy()
        sky_radius = self._sky_radius.repeat(xyz.shape[0],1).detach().cpu().numpy()
        sky_gauss_center = self._sky_gauss_center.repeat(xyz.shape[0],1).cpu().numpy()
        sky_angles = torch.zeros((self.get_xyz.shape[0], 2), device="cuda", requires_grad=True)
        sky_angles[sky_pts_mask] = self._sky_angles
        sky_angles = sky_angles.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if torch.sum(self._is_sky) > 0:
            attributes = np.concatenate((xyz, normals, albedo, opacities, scale, rotation, is_sky, sky_radius, sky_gauss_center, sky_angles), axis=1)

        else:
            attributes = np.concatenate((xyz, normals, albedo, opacities, scale, rotation, is_sky), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        albedo = np.zeros((xyz.shape[0], 3))
        albedo[:, 0] = np.asarray(plydata.elements[0]["albedo_0"])
        albedo[:, 1] = np.asarray(plydata.elements[0]["albedo_1"])
        albedo[:, 2] = np.asarray(plydata.elements[0]["albedo_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        is_sky = np.asarray(plydata.elements[0]["is_sky"], dtype=bool)[..., np.newaxis]
        
        sky_radius = np.asarray(plydata.elements[0]["sky_radius"])[0]
        sky_gauss_center_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sky_gauss_center_")]
        sky_gauss_center = np.zeros((xyz.shape[0], len(sky_gauss_center_names)))
        for idx, attr_name in enumerate(sky_gauss_center_names):
            sky_gauss_center[:, idx] = np.asarray(plydata.elements[0][attr_name])
        sky_gauss_center = sky_gauss_center[0]
        sky_angles_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sky_angles_")]
        sky_angles = np.zeros((xyz.shape[0], len(sky_angles_names)))
        for idx, attr_name in enumerate(sky_angles_names):
            sky_angles[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # xyz = xyz[is_sky.squeeze() == 0] # TODO: So weird to have this here?

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo[~(is_sky.squeeze())], dtype=torch.float, device="cuda").requires_grad_(True))

        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._is_sky = torch.tensor(is_sky, dtype=torch.bool, device="cuda")
        self._sky_radius = nn.Parameter(torch.tensor(sky_radius, dtype=torch.float, device="cuda").requires_grad_(True))
        self._sky_gauss_center = nn.Parameter(torch.tensor(sky_gauss_center, dtype=torch.float, device="cuda"))
        self._sky_angles = torch.tensor(sky_angles[is_sky.squeeze()], dtype=torch.float, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["sky_sh", "mlp", "embeddings", "sky_radius"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, mask_fg, mask_sky):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["uncertainty_model", "mlp", "embeddings", "sky_radius"]:
                continue
            if group["name"] in ["xyz", "albedo"]:
                mask_prune = mask_fg
            elif group["name"] == "sky_angles":
                mask_prune = mask_sky
            else:
                mask_prune = mask
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask_prune]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask_prune]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask_prune].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask_prune].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        valid_fg_points_mask = valid_points_mask[~self.get_is_sky.squeeze()]
        valid_sky_points_mask = valid_points_mask[self.get_is_sky.squeeze()]
        optimizable_tensors = self._prune_optimizer(valid_points_mask, valid_fg_points_mask, valid_sky_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._albedo = optimizable_tensors["albedo"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._sky_angles = optimizable_tensors["sky_angles"]
        self._is_sky = self._is_sky[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["uncertainty_model", "mlp", "embeddings", "sky_radius"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_albedo, new_opacities, new_scaling, new_rotation, new_is_sky, new_sky_angles):
        d = {"albedo": new_albedo,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        if new_sky_angles is not None:
            d.update({"sky_angles": new_sky_angles})
        if new_xyz is not None:
            d.update({"xyz": new_xyz})
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        if "xyz" in optimizable_tensors.keys():
            self._xyz = optimizable_tensors["xyz"]
        self._albedo = optimizable_tensors["albedo"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if "sky_angles" in optimizable_tensors.keys():
            self._sky_angles = optimizable_tensors["sky_angles"] # TODO: Take a look at this later
        self._sky_angles = optimizable_tensors["sky_angles"]
        self._is_sky = torch.cat((self._is_sky, new_is_sky), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        selected_fg_pts_mask = selected_pts_mask[~self.get_is_sky.squeeze()] # Foreground points
        selected_sky_pts_mask = selected_pts_mask[self.get_is_sky.squeeze()] # Sky points

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_albedo = self._albedo[selected_fg_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_is_sky = self._is_sky[selected_pts_mask].repeat(N,1)

        # Project sampled positions for sky Gaussians on the sphere (idk wtf this is) TODO
        if torch.sum(selected_sky_pts_mask) > 0:
            new_xyz[new_is_sky.squeeze()] = self._sky_gauss_center + self._sky_radius * (new_xyz[new_is_sky.squeeze()] - self._sky_gauss_center)/torch.norm(new_xyz[new_is_sky.squeeze()] - self._sky_gauss_center, dim=1)[..., None]
            new_sky_angles = cartesian_to_polar_coord(new_xyz[new_is_sky.squeeze()], self._sky_gauss_center.squeeze())
        else:
            new_sky_angles = None

        self.densification_postfix(new_xyz[~(new_is_sky.squeeze())], new_albedo, new_opacity, new_scaling, new_rotation, new_is_sky, new_sky_angles) # TODO: Take a look at this later

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        selected_fg_pts_mask = selected_pts_mask[~self.get_is_sky.squeeze()]
        selected_sky_pts_mask = selected_pts_mask[self.get_is_sky.squeeze()]
        new_xyz = self.get_xyz[selected_pts_mask & ~self.get_is_sky.squeeze()] # TODO: Take a look at this later
        new_albedo = self._albedo[selected_fg_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_sky_angles = self._sky_angles[selected_sky_pts_mask]
        new_is_sky = self._is_sky[selected_pts_mask]
        self.densification_postfix(new_xyz, new_albedo, new_opacities, new_scaling, new_rotation, new_is_sky, new_sky_angles)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1