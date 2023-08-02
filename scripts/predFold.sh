

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_1/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/Exvivo \
--name                 fold_1 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_2/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/Exvivo \
--name                 fold_2 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_3/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/Exvivo \
--name                 fold_3 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_4/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/Exvivo \
--name                 fold_4 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_5/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/Exvivo \
--name                 fold_5 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_6/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/Exvivo \
--name                 fold_6 \
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










python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/MnM/CV/fold_1/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/MnM \
--name                 fold_1 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/MnM/CV/fold_2/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/MnM \
--name                 fold_2 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/MnM/CV/fold_3/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/MnM \
--name                 fold_3 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/MnM/CV/fold_4/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/MnM \
--name                 fold_4 \
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










python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi/fold_1/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/LGE/96x96x24_mi \
--name                 fold_1 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi/fold_2/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/LGE/96x96x24_mi \
--name                 fold_2 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi/fold_3/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/LGE/96x96x24_mi \
--name                 fold_3 \
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

python /home/maxi/Segmentation/Code/3Dseg/pred.py \
--root_path            /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi/fold_4/val \
--results_dir          /home/maxi/Segmentation/Results_paper2/LGE/96x96x24_mi \
--name                 fold_4 \
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
