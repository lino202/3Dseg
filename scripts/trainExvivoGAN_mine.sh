#!/bin/bash

python /home/maxi/Segmentation/Code/3Dseg/trainGAN.py \
--root_path       /home/maxi/Segmentation/Data/seg3D_exvivo/ph_checked_onlymine \
--results_dir     /home/maxi/Segmentation/Results/seg3D_exvivo_GAN/ph_checked_onlymine_GAN \
--name            100_100_GAN_128 \
--phase           train \
--input_nc        1 \
--output_nc       1 \
--nfl             64 \
--num_downs       7 \
--n_epochs        100 \
--n_epochs_decay  100 \
--gan             \
--gan_mode        lsgan \
--load_size_h     128 \
--load_size_w     128 \
--load_size_d     128 \
--save_epoch_freq 25 \
--batch_size      1 \
