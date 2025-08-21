#!/bin/bash
# LumiGauss: Training, Testing, and Rendering

# # #########################
# # Dataset: ST
# # #########################
RESOLUTION=2

ST_SOURCE_PATH="/home/islamoglu/maximum-effort/dataset/nerf-osr/st_colmap/undistorted"
ST_OUTPUT_PATH="output/st"
ST_EVAL_PATH="/home/islamoglu/maximum-effort/dataset/nerf-osr/st_colmap/eval_files"
# mkdir -p "$ST_OUTPUT_PATH"

# python train.py -s="$ST_SOURCE_PATH" -m="$ST_OUTPUT_PATH" --consistency_loss_lambda_init 10.0 -r $RESOLUTION --with_mlp

# Test with ground truth environment map
python test_gt_env_map.py -s="$ST_SOURCE_PATH" -m="$ST_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --test_config $ST_EVAL_PATH --eval_file $ST_SOURCE_PATH/st_split.csv

# Render rotating environment map
python render_rotate_envmap.py -s="$ST_SOURCE_PATH" -m="$ST_OUTPUT_PATH" -r $RESOLUTION --with_mlp \
  --envmaps example_envmaps --viewpoints 12-04_18_00_DSC_0483.jpg 20-04_18_00_DSC_1473.jpg

# Render using an appearance list
python render_simple.py -s="$ST_SOURCE_PATH" -m="$ST_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --only_from_appearance_list --appearance_list /home/islamoglu/maximum-effort/lumigauss/example_test_configs/st/appearance_list.txt

sleep 10
# #########################
# Dataset: LWP
# #########################
RESOLUTION=2

LWP_SOURCE_PATH="/home/islamoglu/maximum-effort/dataset/nerf-osr/lwp_colmap/undistorted"
LWP_OUTPUT_PATH="output/lwp"
LWP_EVAL_PATH="/home/islamoglu/maximum-effort/dataset/nerf-osr/lwp_colmap/eval_files"
# mkdir -p "$LWP_OUTPUT_PATH"

# python train.py -s="$LWP_SOURCE_PATH" -m="$LWP_OUTPUT_PATH" --consistency_loss_lambda_init 10.0 -r $RESOLUTION --with_mlp

# # Test with ground truth environment map
python test_gt_env_map.py -s="$LWP_SOURCE_PATH" -m="$LWP_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --test_config $LWP_EVAL_PATH --eval_file $LWP_SOURCE_PATH/lwp_split.csv

# Render rotating environment map
python render_rotate_envmap.py -s="$LWP_SOURCE_PATH" -m="$LWP_OUTPUT_PATH" -r $RESOLUTION --with_mlp \
  --envmaps example_envmaps --viewpoints 26-04_17_50_DSC_2205.jpg 25-04_12_30_DSC_1976.jpg

# Render using an appearance list
python render_simple.py -s="$LWP_SOURCE_PATH" -m="$LWP_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --only_from_appearance_list --appearance_list /home/islamoglu/maximum-effort/lumigauss/example_test_configs/lwp/appearance_list.txt

sleep 10
# ##########################
# # Dataset: LK2
# ##########################
RESOLUTION=2

LK2_SOURCE_PATH="/home/islamoglu/maximum-effort/dataset/nerf-osr/lk2_colmap/undistorted"
LK2_OUTPUT_PATH="output/lk2"
LK2_EVAL_PATH="/home/islamoglu/maximum-effort/dataset/nerf-osr/lk2_colmap/eval_files"
# mkdir -p "$LK2_OUTPUT_PATH"

# python train.py -s="$LK2_SOURCE_PATH" -m="$LK2_OUTPUT_PATH" --consistency_loss_lambda_init 10.0 -r $RESOLUTION --with_mlp

# # Test with ground truth environment map
python test_gt_env_map.py -s="$LK2_SOURCE_PATH" -m="$LK2_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --test_config $LK2_EVAL_PATH --eval_file $LK2_SOURCE_PATH/lk2_split.csv

# Render rotating environment map
python render_rotate_envmap.py -s="$LK2_SOURCE_PATH" -m="$LK2_OUTPUT_PATH" -r $RESOLUTION --with_mlp \
  --envmaps example_envmaps --viewpoints 12-04_10_00_DSC_0359.jpg 07-04_17_30_DSC_0089.jpg

# Render using an appearance list
python render_simple.py -s="$LK2_SOURCE_PATH" -m="$LK2_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
  --only_from_appearance_list --appearance_list /home/islamoglu/maximum-effort/lumigauss/example_test_configs/lk2/appearance_list.txt

sleep 10
#########################
# Dataset: TREVI
#########################
# RESOLUTION=2

# TREVI_SOURCE_PATH="data/trevi_fountain/dense"
# TREVI_OUTPUT_PATH="output/trevi"

# mkdir -p "$TREVI_OUTPUT_PATH"

# python train.py -s="$TREVI_SOURCE_PATH" -m="$TREVI_OUTPUT_PATH" --consistency_loss_lambda_init 1.0 -r $RESOLUTION \
#   --with_mlp --iteration 60000 --start_shadowed 30500 --warmup 30000

# # Render rotating environment map
# python render_rotate_envmap.py -s="$TREVI_SOURCE_PATH" -m="$TREVI_OUTPUT_PATH" -r $RESOLUTION --with_mlp \
#   --envmaps example_envmaps --viewpoints 38014238_3404678433.jpg 10011699_6167688545.jpg

# # Render using an appearance list
# python render_simple.py -s="$TREVI_SOURCE_PATH" -m="$TREVI_OUTPUT_PATH" --with_mlp -r $RESOLUTION \
#   --only_from_appearance_list --appearance_list example_test_configs/trevi/appearance_list.txt

# sleep 10