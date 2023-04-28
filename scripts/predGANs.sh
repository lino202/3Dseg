#!/bin/bash


python /home/maxi/Segmentation/Code/3Dseg/predGAN.py \
--root_path       /home/maxi/Segmentation/Data/seg3D_exvivo/ph_checked_onlyStandfor/train_deform \
--results_dir     /home/maxi/Segmentation/Results/seg3D_exvivo_GAN/ph_checked_onlyStandfor_GAN \
--name            100_100_GAN_128 \
--phase           pred \
--input_nc        1 \
--output_nc       1 \
--nfl             64 \
--num_downs       7 \
--load_size_h     128 \
--load_size_w     128 \
--load_size_d     128 \
--load_filename   latest_net_G \
