#!/bin/bash

for i in $(seq 0 6)
do
	echo "Training fold ${i}!!"
	python /home/maxi/Segmentation/Code/3Dseg/train.py \
	--root_path        /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV_base_deform/fold_${i} \
	--results_dir      /home/maxi/Segmentation/Results_paper2/Exvivo \
	--name             base_deform_fold_${i} \
	--display_env      exvivo_fold_${i}_base_deform \
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
done



for i in $(seq 0 4)
do
	echo "Training fold ${i}!!"
	python /home/maxi/Segmentation/Code/3Dseg/train.py \
	--root_path        /home/maxi/Segmentation/Data_paper2/MnM/CV_base_deform/fold_${i} \
	--results_dir      /home/maxi/Segmentation/Results_paper2/MnM \
	--name             base_deform_fold_${i} \
	--display_env      mnm_fold_${i}_base_deform \
	--phase            train \
	--input_nc         1 \
	--output_nc        4 \
	--nfl              64 \
	--num_downs        5 \
	--lr               0.0002 \
	--n_epochs         125 \
	--n_epochs_decay   125 \
	--loss             CE \
	--patch_size       160 160 20 \
	--batch_size       4 \
	--norm             instance \
	--dataaug
done




for i in $(seq 0 4)
do
	echo "Training fold ${i}!!"
	python /home/maxi/Segmentation/Code/3Dseg/train.py \
	--root_path       /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi_base_deform/fold_${i} \
	--results_dir     /home/maxi/Segmentation/Results_paper2/LGE/96x96x24_mi \
	--name            base_deform_fold_${i} \
	--display_env     lge_mi_fold_${i}_base_deform \
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
done
