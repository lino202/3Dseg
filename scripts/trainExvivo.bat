@REM @ECHO OFF
call conda activate torchTopo

python D:/Segmentation/Code/3Dseg/train.py ^
--root_path        D:/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_0 ^
--results_dir      D:/Segmentation/Results_paper2/Exvivo ^
--name             fold_0 ^
--display_env      fold_0 ^
--phase            train ^
--input_nc         1 ^
--output_nc        2 ^
--nfl              64 ^
--num_downs        5 ^
--n_epochs         10 ^
--n_epochs_decay   10 ^
--loss             CE ^
--patch_size       128 128 128 ^
--batch_size       3 ^
--norm             instance ^
--dataaug

python D:/Segmentation/Code/3Dseg/train.py ^
--root_path        D:/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_1 ^
--results_dir      D:/Segmentation/Results_paper2/Exvivo ^
--name             fold_1 ^
--display_env      fold_1 ^
--phase            train ^
--input_nc         1 ^
--output_nc        2 ^
--nfl              64 ^
--num_downs        5 ^
--n_epochs         10 ^
--n_epochs_decay   10 ^
--loss             CE ^
--patch_size       128 128 128 ^
--batch_size       3 ^
--norm             instance ^
--dataaug


python D:/Segmentation/Code/3Dseg/train.py ^
--root_path        D:/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_2 ^
--results_dir      D:/Segmentation/Results_paper2/Exvivo ^
--name             fold_2 ^
--display_env      fold_2 ^
--phase            train ^
--input_nc         1 ^
--output_nc        2 ^
--nfl              64 ^
--num_downs        5 ^
--n_epochs         10 ^
--n_epochs_decay   10 ^
--loss             CE ^
--patch_size       128 128 128 ^
--batch_size       3 ^
--norm             instance ^
--dataaug