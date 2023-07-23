

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_0/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/Exvivo \
--name                 fold_0 \
--phase                pred \
--input_nc             1 \
--output_nc            2 \
--nfl                  64 \
--num_downs            5 \
--patch_size           128 128 128 \
--load_filename        latest_net \
--norm                 instance \
--phThres              0.5 \
--phConstruction       N \
--priorName            PRIOR_EXVIVO



python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            D:/Segmentation/Data_paper2/MnM/CV/fold_0/val \
--results_dir          D:/Segmentation/Results_paper2/MnM \
--name                 fold_0 \
--phase                pred \
--input_nc             1 \
--output_nc            4 \
--nfl                  64 \
--num_downs            5 \
--patch_size           160 160 20 \
--load_filename        latest_net \
--norm                 instance \
--phThres              0.5 \
--phConstruction       N \
--priorName            PRIOR_CINE



python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            D:/Segmentation/Data_paper2/LGE/96x96x24/CV_mi/fold_0/val \
--results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi \
--name                 fold_0 \
--phase                pred \
--input_nc             1 \
--output_nc            4 \
--nfl                  64 \
--num_downs            5 \
--patch_size           96 96 24 \
--load_filename        latest_net \
--norm                 instance \
--phThres              0.5 \
--phConstruction       N \
--priorName            PRIOR_LGE