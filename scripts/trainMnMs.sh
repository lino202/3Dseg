#!/bin/bash

python /home/maxi/Segmentation/Code/3Dseg/train.py \
--root_path       /home/maxi/Segmentation/Data/seg3D_MnMs/128x128_ED \
--results_dir     /home/maxi/Segmentation/Results/seg3D_MnMs/128x128_ED \
--name            50_50_CE_128_1_deform \
--phase           train \
--input_nc        1 \
--output_nc       4 \
--nfl             64 \
--num_downs       7 \
--n_epochs        50 \
--n_epochs_decay  50 \
--loss            CE \
--load_size_h     128 \
--load_size_w     128 \
--load_size_d     128 \
--save_epoch_freq 25 \
--batch_size      1 \
