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

import os
import numpy as np
import torch
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, UncertaintyParams
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.vis_utils import apply_depth_colormap, colormap
from utils.depth_utils import depth_to_normal

def scale_grads(values, scale):
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values

def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image

def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, unc: UncertaintyParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, uncertainty_model : Optional[UncertaintyModel] = None):
    iteration = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, uncertainty_config=unc)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, iteration) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    # cameras = scene.getTrainCameras() + scene.getValidationCameras() + scene.getTestCameras()
    cameras = scene.getTrainCameras() + scene.getTestCameras()
    for idx, camera in enumerate(cameras):
        camera.idx = idx
    
    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_uncertainty_for_log = 0.0

    progress_bar = tqdm(range(iteration, opt.iterations), desc="Training progress")
    iteration += 1 # Gaussian Splatting is 1-indexed
    for iter in range(iteration, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iter % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))
        viewpoint_cam_id = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_cam_id]
        

        if (iter - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] #TODO: add image_toned in here somehow

        image = rendering[:3, :, :]

        # image_toned  = None # TODO: add image_toned rendering[:3, :, :] probably
        gt_image = viewpoint_cam.original_image.to(dataset.data_device)
        uncertainty_loss = 0
        uncertainty_metrics = {}
        loss_mult = 1.0

        if gaussians.uncertainty_model is not None:
            del loss_mult
            uncertainty_loss, uncertainty_metrics, loss_mult = gaussians.uncertainty_model.get_loss(gt_image, image.detach(), _cache_entry=('train', viewpoint_cam_id))
            # uncertainty_warmup_iters: int = 0
            # uncertainty_warmup_start: int = 0

            loss_mult = (loss_mult > 1).to(dtype=loss_mult.dtype)

            if iter < unc.uncertainty_warmup_start:
                loss_mult = 1
            elif iter < unc.uncertainty_warmup_start + unc.uncertainty_warmup_iters:
                p = (iter - unc.uncertainty_warmup_start) / unc.uncertainty_warmup_iters
                loss_mult = 1 + p * (loss_mult - 1)
            if unc.uncertainty_center_mult:
                loss_mult = loss_mult.sub(loss_mult.mean() - 1).clamp(0, 2)
            if unc.uncertainty_scale_grad:
                image = scale_grads(image, loss_mult)
                # image_toned = scale_grads(image_toned, loss_mult)
                loss_mult = 1


        Ll1 = torch.nn.functional.l1_loss(image, gt_image, reduction='none')
        ssim_value = ssim(image, gt_image, size_average=False)
        # use L1 loss for the transformed image if using decoupled appearance
        if dataset.use_decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)

        # Detach uncertainty loss if in protected iter after opacity reset
        if gaussians.uncertainty_model is not None:

            last_densify_iter = min(iter, opt.densify_until_iter - 1)
            last_dentify_iter = (last_densify_iter // opt.opacity_reset_interval) * opt.opacity_reset_interval
            if iter < last_dentify_iter + unc.uncertainty_protected_iters:
                # Keep track of max radii in image-space for pruning
                try:
                    uncertainty_loss = uncertainty_loss.detach()  # type: ignore
                except AttributeError:
                    pass

        rgb_loss = (1.0 - opt.lambda_dssim) * (Ll1 * loss_mult).mean() + opt.lambda_dssim * ((1.0 - ssim_value) * loss_mult).mean()

        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        # edge aware regularization is not really helpful so we disable it
        # distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)
        distortion_loss = distortion_map.mean()
        
        # depth normal consistency
        depth = rendering[6, :, :]
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
        
        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        normal_error = 1 - (render_normal_world * depth_normal).sum(dim=0)
        depth_normal_loss = normal_error.mean()
        
        lambda_distortion = opt.lambda_distortion if iter >= opt.distortion_from_iter else 0.0
        lambda_depth_normal = opt.lambda_depth_normal if iter >= opt.depth_normal_from_iter else 0.0

        # Final loss
        loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion + uncertainty_loss
        loss.backward()

        iter_end.record()

        is_save_images = False # default to not save images
        if is_save_images and (iteration % opt.densification_interval == 0):
            with torch.no_grad():
                eval_cam = allCameras[random.randint(0, len(allCameras) -1)]
                
                rendering = render(eval_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)["render"]
                image = rendering[:3, :, :]
                transformed_image = L1_loss_appearance(image, eval_cam.original_image.cuda(), gaussians, eval_cam.idx, return_transformed_image=True)
                
                normal = rendering[3:6, :, :]
                normal = torch.nn.functional.normalize(normal, p=2, dim=0)
                
            # transform to world space
            c2w = (eval_cam.world_view_transform.T).inverse()
            normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
            normal = normal2.reshape(3, *normal.shape[1:])
            normal = (normal + 1.) / 2.
            
            depth = rendering[6, :, :]
            depth_normal, _ = depth_to_normal(eval_cam, depth[None, ...])
            depth_normal = (depth_normal + 1.) / 2.
            depth_normal = depth_normal.permute(2, 0, 1)
            
            gt_image = eval_cam.original_image.cuda()
            
            depth_map = apply_depth_colormap(depth[..., None], rendering[7, :, :, None], near_plane=None, far_plane=None)
            depth_map = depth_map.permute(2, 0, 1)
            
            accumlated_alpha = rendering[7, :, :, None]
            colored_accum_alpha = apply_depth_colormap(accumlated_alpha, None, near_plane=0.0, far_plane=1.0)
            colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)
            
            distortion_map = rendering[8, :, :]
            distortion_map = colormap(distortion_map.detach().cpu().numpy()).to(normal.device)
        
            row0 = torch.cat([gt_image, image, depth_normal, normal], dim=2)
            row1 = torch.cat([depth_map, colored_accum_alpha, distortion_map, transformed_image], dim=2)
            
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0, 1)
            
            os.makedirs(f"{dataset.model_path}/log_images", exist_ok = True)
            torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/log_images/{iter}.jpg")

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iter % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iter == opt.iterations:
                progress_bar.close()            

            # Log and save
            training_report(tb_writer, iter, Ll1.mean(), loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            if (iter in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iter))
                scene.save(iter)

            # Densification
            if iter < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iter > opt.densify_from_iter and iter % opt.densification_interval == 0:
                    size_threshold = 20 if iter > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iter % opt.opacity_reset_interval == 0 or (dataset.white_background and iter == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iter % 100 == 0 and iter > opt.densify_until_iter:
                if iter < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            if iter < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iter in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iter))
                torch.save((gaussians.capture(), iter), scene.model_path + "/chkpnt" + str(iter) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        print("Tensorboard writer created: {}".format(args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    image = rendering[:3, :, :]
                    normal = rendering[3:6, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    up = UncertaintyParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 30_000, 50_000, 70_000, 90_000, 100_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_000, 30_000, 50_000, 70_000, 90_000, 100_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), up.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, uncertainty_model)

    # All done
    print("\nTraining complete.")
