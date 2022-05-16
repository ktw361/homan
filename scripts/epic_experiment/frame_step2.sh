FRAMES=$1

python fit_epic_frame.py \
    --lw_smooth 0 \
    --lw_collision 0.001 \
    --lw_contact 1 \
    --optimize_object_scale 0 \
    --result_root results/epic_frame_gt/step2 \
    --resume results/epic_frame_gt/step1  \
    --interpolation_dir /home/skynet/Zhifan/data/epic_analysis/interpolation \
    --frames_file ${FRAMES}

# --frames_file /home/skynet/Zhifan/data/epic_analysis/clean_tiny_gt.txt
