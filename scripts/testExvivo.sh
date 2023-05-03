#!/bin/bash

python /home/maxi/Segmentation/Code/3Dseg/test.py \
--root_path       /home/maxi/Segmentation/Data/seg3D_exvivo/ph_checked_onlyStandfor \
--results_dir     /home/maxi/Segmentation/Results/seg3D_exvivo/ph_checked_onlyStandfor \
--name            50_50_CE_128_1_deform \
--phase           test \
--input_nc        1 \
--output_nc       2 \
--nfl             64 \
--num_downs       7 \
--load_size_h     128 \
--load_size_w     128 \
--load_size_d     128 \
--load_filename   latest_net \
--phThres         -1 \
--ph              \
--phParallel      \
--priorName        prior_roi \
--res_excel        /home/maxi/Segmentation/Results/seg3D_exvivo/ph_checked_onlyStandfor/50_50_CE_128_1_deform/results.xlsx \
--res_params_name  results_ph \
--res_excel_indexs gdsc_bg gdsc_myo hd_bg hd_myo be ts
