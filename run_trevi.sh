#!/bin/bash
# LumiGauss: Training, Testing, and Rendering

##########################
# Dataset: TREVI
##########################
# ---------- static paths ----------
TREVI_SOURCE_PATH="/home/islamoglu/maximum-effort/data/lumigauss/trevi/data"
BASE_OUTPUT_DIR="/home/islamoglu/maximum-effort/data/lumigauss/trevi_new"                # will hold one sub-folder per resolution

# ---------- hyper-parameters ----------
SLEEP_SECONDS=60                        # pause between resolutions
ITERATIONS=60000
START_SHADOWED=30500
WARMUP=30000
CONSISTENCY_LAMBDA=1.0
VIEWPOINTS="38014238_3404678433.jpg 10011699_6167688545.jpg"

# ---------- main sweep ----------
for RESOLUTION in 2 4 8; do
  RUN_DIR="${BASE_OUTPUT_DIR}/trevi_res${RESOLUTION}"
  mkdir -p "$RUN_DIR"

  echo "====================  RES $RESOLUTION  ===================="
  echo "Output directory: $RUN_DIR"

  # ---------- training ----------
  python train.py \
    -s "$TREVI_SOURCE_PATH" \
    -m "$RUN_DIR" \
    --consistency_loss_lambda_init "$CONSISTENCY_LAMBDA" \
    -r "$RESOLUTION" \
    --with_mlp \
    --iteration "$ITERATIONS" \
    --start_shadowed "$START_SHADOWED" \
    --warmup "$WARMUP"

  # ---------- rotating env-map render ----------
  python render_rotate_envmap.py \
    -s "$TREVI_SOURCE_PATH" \
    -m "$RUN_DIR" \
    -r "$RESOLUTION" \
    --with_mlp \
    --envmaps example_envmaps \
    --viewpoints $VIEWPOINTS

  # ---------- appearance-list render ----------
  python render_simple.py \
    -s "$TREVI_SOURCE_PATH" \
    -m "$RUN_DIR" \
    --with_mlp \
    -r "$RESOLUTION" \
    --only_from_appearance_list \
    --appearance_list example_test_configs/trevi/appearance_list_trevi.txt

  echo "Resolution $RESOLUTION finished — sleeping ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
done

echo "All resolutions completed successfully."

