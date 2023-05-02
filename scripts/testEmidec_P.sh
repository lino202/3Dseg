#!/bin/bash

python /home/maxi/Segmentation/Code/3Dseg/test.py \
--root_path       /home/maxi/Segmentation/Data/seg3D_emidec/cropped_test_P \
--results_dir     /home/maxi/Segmentation/Results/seg3D_emidec/cropped \
--name            50_50_CE_128_1_deform \
--phase           test \
--input_nc        1 \
--output_nc       4 \
--nfl             64 \
--num_downs       7 \
--load_size_h     128 \
--load_size_w     128 \
--load_size_d     128 \
--load_filename   latest_net \
--phThres         0.5 \
--phParallel      \
--priorName        prior_LGE_emidec \
--res_excel        /home/maxi/Segmentation/Results/seg3D_emidec/cropped/50_50_CE_128_1_deform/results.xlsx \
--res_params_name  results_baseline_P \
--res_excel_indexs gdsc_bg gdsc_lv gdsc_myo gdsc_mi hd_bg hd_lv hd_myo hd_mi be ts
