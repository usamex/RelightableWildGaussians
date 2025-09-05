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
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, depth_loss_gaussians, envl_sh_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, UncertaintyParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def scale_grads(values, scale):
    """Scale gradients for uncertainty-based weighting"""
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values

def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, uncertainty_opt: UncertaintyParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(uncertainty_config=uncertainty_opt, model_config=dataset, length_train_cameras=397)
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        checkpoint_dir = os.path.dirname(checkpoint)
        if gaussians.with_mlp:
            gaussians.mlp.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_mlp" + str(first_iter) + ".pth")))
            gaussians.embedding.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_embedding" + str(first_iter) + ".pth")))
        else:
            gaussians.env_params.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_env" + str(first_iter) + ".pth")))
        gaussians.optimizer_env.load_state_dict(torch.load(os.path.join(checkpoint_dir, "chkpnt_optimizer_env" + str(first_iter) + ".pth")))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_logs = {
        "base_loss": 0.0,
        "total_loss": 0.0,
        "distortion_loss": 0.0,
        "normal_loss": 0.0,
        "uncertainty_loss": 0.0,
        "sky_loss": 0.0,
        "envlight_loss": 0.0,
        # "depth_loss": 0.0,
        "points": 0.0
    }


    # Prepare a look up table for the appearance embedding
    appearance_lut = {}
    for i, s in enumerate(scene.getTrainCameras().copy()):
        appearance_lut[s.image_name] = i
    with open(os.path.join(scene.model_path, "appearance_lut.json"), "w") as outfile: 
        json.dump(appearance_lut, outfile)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))
        viewpoint_cam_id = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_cam_id]
        
        gt_image = viewpoint_cam.original_image.to("cuda")
        sky_mask = viewpoint_cam.sky_mask.expand_as(gt_image).to("cuda")
        occluders_mask = viewpoint_cam.occluders_mask.expand_as(gt_image).to("cuda")

        # Get SH coefficients of environment lighting for current training image
        emb_idx = appearance_lut[viewpoint_cam.image_name]
        envlight_sh, sky_sh = gaussians.compute_env_sh(emb_idx)
        envlight_sh_rand_noise = torch.randn_like(envlight_sh)*0.025 # Adding random noise to the environment lighting SH coefficients just like in NerfOSR
        # Get environment lighting object for the current training image
        gaussians.envlight.set_base(envlight_sh + envlight_sh_rand_noise)

        # Apply mask TODO: I'm not sure if we are doing it correctly
        # if sky_mask is not None:
        #     image = scale_grads(image, sky_mask)

        render_pkg = render(viewpoint_cam, gaussians, gaussians.envlight, sky_sh, dataset.sky_sh_degree, pipe, background, debug=False, fix_sky=dataset.fix_sky, specular=dataset.specular)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        diff_col, spec_col = render_pkg["diffuse_color"], render_pkg["specular_color"]

        # Uncertainty computation
        uncertainty_loss = 0
        uncertainty_metrics = {}
        loss_mult = 1.0
        
        if gaussians.uncertainty_model is not None:
            del loss_mult
            # Compute uncertainty loss and multipliers
            uncertainty_loss, uncertainty_metrics, loss_mult = gaussians.uncertainty_model.get_loss(
                gt_image, 
                image.detach(), 
                _cache_entry=('train', viewpoint_cam_id)
            )
            
            # Apply uncertainty warmup logic
            loss_mult = (loss_mult > 1).to(dtype=loss_mult.dtype)
            
            if iteration < uncertainty_opt.uncertainty_warmup_start:
                loss_mult = 1
            elif iteration < uncertainty_opt.uncertainty_warmup_start + uncertainty_opt.uncertainty_warmup_iters:
                p = (iteration - uncertainty_opt.uncertainty_warmup_start) / uncertainty_opt.uncertainty_warmup_iters
                loss_mult = 1 + p * (loss_mult - 1)
                
            if uncertainty_opt.uncertainty_center_mult:
                loss_mult = loss_mult.sub(loss_mult.mean() - 1).clamp(0, 2)
            
            if uncertainty_opt.uncertainty_scale_grad:
                image = scale_grads(image, loss_mult)
                loss_mult = 1
        
        # Compute losses with uncertainty weighting
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image, size_average=False)
        
        # Apply loss multipliers for uncertainty weighting
        base_loss = (1.0 - opt.lambda_dssim) * (Ll1 * loss_mult).mean() + opt.lambda_dssim * ((1.0 - ssim_value) * loss_mult).mean()
        sky_loss = opt.lambda_sky_brdf * (l1_loss(diff_col, torch.zeros_like(diff_col), mask=(1-sky_mask)) + l1_loss(spec_col, torch.zeros_like(spec_col), mask=(1-sky_mask)))

        # regularization DO NOT TOUCH
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # Envlight regularization
        envl_loss = envl_sh_loss(envlight_sh, dataset.envlight_sh_degree)

        depth_loss_sky_gauss = 0
        # Depth regularization
        # if iteration > opt.reg_sky_gauss_depth_from_iter and opt.lambda_sky_gauss > 0:
        #     sky_gaussians_mask = gaussians.get_is_sky.squeeze()
        #     gaussians_depth = gaussians.get_depth(viewpoint_cam)
        #     sky_gaussians_depth = gaussians_depth[(sky_gaussians_mask) & (visibility_filter)]
        #     avg_depth_sky_gauss = torch.mean(sky_gaussians_depth)
        #     avg_depth_non_sky_gauss = torch.mean(gaussians_depth[(~sky_gaussians_mask) & (visibility_filter)]).detach()
        #     depth_loss_sky_gauss = opt.lambda_sky_gauss * depth_loss_gaussians(avg_depth_sky_gauss, avg_depth_non_sky_gauss)

        # Detach uncertainty loss if in protected iterations after opacity reset
        if gaussians.uncertainty_model is not None:
            last_densify_iter = min(iteration, opt.densify_until_iter - 1)
            last_dentify_iter = (last_densify_iter // opt.opacity_reset_interval) * opt.opacity_reset_interval
            if iteration < last_dentify_iter + uncertainty_opt.uncertainty_protected_iters:
                try:
                    uncertainty_loss = uncertainty_loss.detach()
                except AttributeError:
                    pass

        # Total loss with uncertainty
        total_loss = base_loss + dist_loss + normal_loss + uncertainty_loss + sky_loss + envl_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_logs["base_loss"] = 0.4 * base_loss.item() + 0.6 * ema_logs["base_loss"]
            ema_logs["distortion_loss"] = 0.4 * dist_loss.item() + 0.6 * ema_logs["distortion_loss"]
            ema_logs["normal_loss"] = 0.4 * normal_loss.item() + 0.6 * ema_logs["normal_loss"]
            if isinstance(uncertainty_loss, torch.Tensor):
                ema_logs["uncertainty_loss"] = 0.4 * uncertainty_loss.item() + 0.6 * ema_logs["uncertainty_loss"]

            ema_logs["total_loss"] = 0.4 * total_loss.item() + 0.6 * ema_logs["total_loss"]
            ema_logs["sky_loss"] = 0.4 * sky_loss.item() + 0.6 * ema_logs["sky_loss"]
            ema_logs["envlight_loss"] = 0.4 * envl_loss.item() + 0.6 * ema_logs["envlight_loss"]
            # ema_logs["depth_loss"] = 0.4 * depth_loss_sky_gauss.item() + 0.6 * ema_logs["depth_loss"]

            if iteration % 10 == 0:
                loss_dict = {
                    "BaseLoss": f"{ema_logs['base_loss']:.{5}f}",
                    "TotalLoss": f"{ema_logs['total_loss']:.{5}f}",
                    "Distortion": f"{ema_logs['distortion_loss']:.{5}f}",
                    "Normal": f"{ema_logs['normal_loss']:.{5}f}",
                    "Uncertainty": f"{ema_logs['uncertainty_loss']:.{5}f}",
                    "Sky": f"{ema_logs['sky_loss']:.{5}f}",
                    "Envlight": f"{ema_logs['envlight_loss']:.{5}f}",
                    # "Depth": f"{ema_logs['depth_loss']:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_logs["distortion_loss"], iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_logs["normal_loss"], iteration)
                if isinstance(uncertainty_loss, torch.Tensor):
                    tb_writer.add_scalar('train_loss_patches/uncertainty_loss', ema_logs["uncertainty_loss"], iteration)
                    # Log uncertainty metrics
                    for key, value in uncertainty_metrics.items():
                        tb_writer.add_scalar(f'uncertainty_metrics/{key}', value, iteration)

            training_report(tb_writer, iteration, Ll1.mean(), base_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_logs["total_loss"]
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

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
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs): # TODO: Change this
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
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
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

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
            tb_writer.add_histogram("scene/roughness_histogram", scene.gaussians.get_roughness, iteration)
            tb_writer.add_histogram("scene/metalness_histogram", scene.gaussians.get_metalness, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
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
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 50_000, 70_000, 80_000, 90_000, 110_000, 130_000, 150_000, 170_000, 190_000, 200_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 50_000, 70_000, 80_000, 90_000, 110_000, 130_000, 150_000, 170_000, 190_000, 200_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), up.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
