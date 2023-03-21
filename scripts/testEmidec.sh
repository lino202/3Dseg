#!/bin/bash

python /home/maxi/Segmentation/Code/3Dseg/test.py \
--root_path       /home/maxi/Segmentation/Data/seg3D_emidec/128x128/roi \
--results_dir     /home/maxi/Segmentation/Results/seg3D_emidec/roi \
--name            50_50_CE_128_4_1_test1 \
--phase           test \
--input_nc        1 \
--output_nc       1 \
--nfl             64 \
--num_downs       7 \
--load_size_h     128 \
--load_size_w     128 \
--load_size_d     128 \
--load_filename   latest_net \
# --ph            \
--phThres         -1 \ 
# --phParallel      \  
--priorName        prior_LGE_roi \           
--res_excel        /home/maxi/Segmentation/Results/seg3D_emidec/roi/results.xlsx
--res_params_name  results_baseline
--res_excel_indexs gdsc hd be ts