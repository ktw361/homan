#!/bin/sh
FRAMES=$1
# FRAMES=/home/skynet/Zhifan/data/epic_analysis/clean_tiny_gt.txt

python fit_epic_frame.py \
    --optimize_object_scale 0 \
    --result_root results/epic_frame_gt/step1 \
    --num_initializations 400 \
    --only_missing 1 \
    --interpolation_dir /home/skynet/Zhifan/data/epic_analysis/interpolation \
    --frames_file ${FRAMES}

# --frames_file /home/skynet/Zhifan/data/epic_analysis/clean_tiny_gt.txt
