
for i in $(seq 0 6)
do
        echo "Training fold ${i}!!"
	python /home/maxi/Segmentation/Code/3Dseg/pred.py \
	--root_path            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV_base_deform/fold_${i}/val \
	--results_dir          /home/maxi/Segmentation/Results_paper2/Exvivo \
	--name                 base_deform_fold_${i} \
	--phase                pred \
	--input_nc             1 \
	--output_nc            2 \
	--nfl                  64 \
	--num_downs            5 \
	--patch_size           128 128 128 \
	--load_filename        latest_net \
	--norm                 instance \
	--ph                   \
	--phThres              0.5 \
	--phConstruction       N \
	--priorName            PRIOR_EXVIVO
done



for i in $(seq 0 4)
do
        echo "Training fold ${i}!!"
	python /home/maxi/Segmentation/Code/3Dseg/pred.py \
	--root_path            /home/maxi/Segmentation/Data_paper2/MnM/CV_base_deform/fold_${i}/val \
	--results_dir          /home/maxi/Segmentation/Results_paper2/MnM \
	--name                 base_deform_fold_${i} \
	--phase                pred \
	--input_nc             1 \
	--output_nc            4 \
	--nfl                  64 \
	--num_downs            5 \
	--patch_size           160 160 20 \
	--load_filename        latest_net \
	--norm                 instance \
	--ph                   \
	--phThres              0.5 \
	--phConstruction       N \
	--priorName            PRIOR_CINE
done



for i in $(seq 0 4)
do
        echo "Training fold ${i}!!"
	python /home/maxi/Segmentation/Code/3Dseg/pred.py \
	--root_path            /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi_base_deform/fold_${i}/val \
	--results_dir          /home/maxi/Segmentation/Results_paper2/LGE/96x96x24_mi \
	--name                 base_deform_fold_${i} \
	--phase                pred \
	--input_nc             1 \
	--output_nc            4 \
	--nfl                  64 \
	--num_downs            5 \
	--patch_size           96 96 24 \
	--load_filename        latest_net \
	--norm                 instance \
	--ph                   \
	--phThres              0.5 \
	--phConstruction       N \
	--priorName            PRIOR_LGE
done
