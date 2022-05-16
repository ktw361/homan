#!/bin/sh
FRAMES=$1
# FRAMES=/home/skynet/Zhifan/data/epic_analysis/clean_tiny_gt.txt

python fit_epic_frame.py \
    --interpolation_dir /home/skynet/Zhifan/data/epic_analysis/interpolation \
    --num_initializations 400 \
    --optimize_object_scale 0 \
    --lw_smooth 0 \
    --result_root results/epic_frame_gt/step1 \
    --frames_file ${FRAMES}

    # --only_missing 1 \
# --frames_file /home/skynet/Zhifan/data/epic_analysis/clean_tiny_gt.txt
