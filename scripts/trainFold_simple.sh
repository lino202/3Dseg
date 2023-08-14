#!/bin/bash


python /home/maxi/Segmentation/Code/3Dseg/train.py \
--root_path        /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV_base_deform/fold_0 \
--results_dir      /home/maxi/Segmentation/Results_paper2/Exvivo \
--name             base_deform_fold_0 \
--display_env      exvivo_fold_0_base_deform \
--phase            train \
--input_nc         1 \
--output_nc        2 \
--nfl              64 \
--num_downs        5 \
--lr               0.0002 \
--n_epochs         125 \
--n_epochs_decay   125 \
--loss             CE \
--patch_size       128 128 128 \
--batch_size       3 \
--norm             instance \
--dataaug


python /home/maxi/Segmentation/Code/3Dseg/train.py \
--root_path        /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV_base_deform/fold_4 \
--results_dir      /home/maxi/Segmentation/Results_paper2/Exvivo \
--name             base_deform_fold_4 \
--display_env      exvivo_fold_4_base_deform \
--phase            train \
--input_nc         1 \
--output_nc        2 \
--nfl              64 \
--num_downs        5 \
--lr               0.0002 \
--n_epochs         125 \
--n_epochs_decay   125 \
--loss             CE \
--patch_size       128 128 128 \
--batch_size       3 \
--norm             instance \
--dataaug

#python /home/maxi/Segmentation/Code/3Dseg/train.py \
#--root_path        /home/maxi/Segmentation/Data_paper2/MnM/CV_base_deform/fold_${i} \
#--results_dir      /home/maxi/Segmentation/Results_paper2/MnM \
#--name             base_deform_fold_${i} \
#--display_env      mnm_fold_${i}_base_deform \
#--phase            train \
#--input_nc         1 \
#--output_nc        4 \
#--nfl              64 \
#--num_downs        5 \
#--lr               0.0002 \
#--n_epochs         125 \
#--n_epochs_decay   125 \
#--loss             CE \
#--patch_size       160 160 20 \
#--batch_size       4 \
#--norm             instance \
#--dataaug




python /home/maxi/Segmentation/Code/3Dseg/train.py \
--root_path       /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi_base_deform/fold_1 \
--results_dir     /home/maxi/Segmentation/Results_paper2/LGE/96x96x24_mi \
--name            base_deform_fold_1 \
--display_env     lge_mi_fold_1_base_deform \
--phase           train \
--input_nc        1 \
--output_nc       4 \
--nfl             64 \
--num_downs       5 \
--lr              0.0002 \
--n_epochs        125 \
--n_epochs_decay  125 \
--loss            CE \
--patch_size      96 96 24 \
--batch_size      4 \
--norm            instance \
--dataaug



python /home/maxi/Segmentation/Code/3Dseg/train.py \
--root_path       /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi_base_deform/fold_4 \
--results_dir     /home/maxi/Segmentation/Results_paper2/LGE/96x96x24_mi \
--name            base_deform_fold_4 \
--display_env     lge_mi_fold_4_base_deform \
--phase           train \
--input_nc        1 \
--output_nc       4 \
--nfl             64 \
--num_downs       5 \
--lr              0.0002 \
--n_epochs        125 \
--n_epochs_decay  125 \
--loss            CE \
--patch_size      96 96 24 \
--batch_size      4 \
--norm            instance \
--dataaug
