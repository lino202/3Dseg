#!/bin/bash

python /home/maxi/Segmentation/Code/3Dseg/train.py \
--root_path       /home/maxi/Segmentation/Data/seg3D_exvivo \
--results_dir     /home/maxi/Segmentation/Results/seg3D_exvivo \
--name            50_50_CE_128_2_1 \
--phase           train \
--input_nc        1 \
--output_nc       2 \
--nfl             64 \
--num_downs       7 \
--n_epochs        50 \
--n_epochs_decay  50 \
--loss            CE \
--load_size_h     128 \
--load_size_w     128 \
--load_size_d     128 \
--save_epoch_freq 5 \
--batch_size      1 \
--print_iter      2 \
