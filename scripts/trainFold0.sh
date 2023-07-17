@REM @ECHO OFF
call conda activate torchTopo

python D:/Segmentation/Code/3Dseg/train.py ^
--root_path        D:/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_0 ^
--results_dir      D:/Segmentation/Results_paper2/Exvivo ^
--name             fold_0 ^
--display_env      ex_vivo_fold_0 ^
--phase            train ^
--input_nc         1 ^
--output_nc        2 ^
--nfl              64 ^
--num_downs        5 ^
--lr               0.01 ^
--n_epochs         1 ^
--n_epochs_decay   249 ^
--loss             CE ^
--patch_size       128 128 128 ^
--batch_size       3 ^
--norm             instance ^
--dataaug

python D:/Segmentation/Code/3Dseg/trainGAN.py ^
--root_path        D:/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV/fold_0 ^
--results_dir      D:/Segmentation/Results_paper2/Exvivo_GAN ^
--name             fold_0 ^
--display_env      exvivo_gan_fold_0 ^
--phase            train ^
--input_nc         1 ^
--output_nc        1 ^
--nfl              64 ^
--num_downs        5 ^
--n_layers_D       2 ^
--lr               0.0002 ^
--n_epochs         125 ^
--n_epochs_decay   125 ^
--loss             CE ^
--gan              ^
--gan_mode         lsgan ^
--patch_size       128 128 128 ^
--batch_size       3 ^
--norm             instance ^
--dataaug



python D:/Segmentation/Code/3Dseg/train.py ^
--root_path        D:/Segmentation/Data_paper2/MnM/CV/fold_0 ^
--results_dir      D:/Segmentation/Results_paper2/MnM ^
--name             fold_0 ^
--display_env      mnm_fold_0 ^
--phase            train ^
--input_nc         1 ^
--output_nc        4 ^
--nfl              64 ^
--num_downs        5 ^
--lr               0.01 ^
--n_epochs         1 ^
--n_epochs_decay   249 ^
--loss             CE ^
--patch_size       160 160 20 ^
--batch_size       4 ^
--norm             instance ^
--dataaug


python D:/Segmentation/Code/3Dseg/trainGAN.py ^
--root_path        D:/Segmentation/Data_paper2/MnM/CV/fold_0 ^
--results_dir      D:/Segmentation/Results_paper2/MnM_GAN ^
--name             fold_0 ^
--display_env      mnm_gan_fold_0 ^
--phase            train ^
--input_nc         1 ^
--output_nc        1 ^
--nfl              64 ^
--num_downs        5 ^
--n_layers_D       3 ^
--lr               0.0002 ^
--n_epochs         125 ^
--n_epochs_decay   125 ^
--loss             CE ^
--gan              ^
--gan_mode         lsgan ^
--patch_size       160 160 20 ^
--batch_size       4 ^
--norm             instance ^
--dataaug



python D:/Segmentation/Code/3Dseg/train.py ^
--root_path       D:/Segmentation/Data_paper2/LGE/96x96x24/CV_mi/fold_0 ^
--results_dir     D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
--name            fold_0 ^
--display_env     lge_mi_fold_0 ^
--phase           train ^
--input_nc        1 ^
--output_nc       4 ^
--nfl             64 ^
--num_downs       5 ^
--lr              0.01 ^
--n_epochs        1 ^
--n_epochs_decay  249 ^
--loss            CE ^
--patch_size      96 96 24 ^
--batch_size      4 ^
--norm            instance ^
--dataaug



python D:/Segmentation/Code/3Dseg/trainGAN.py ^
--root_path       D:/Segmentation/Data_paper2/LGE/96x96x24/CV_mi/fold_0 ^
--results_dir     D:/Segmentation/Results_paper2/LGE_GAN/96x96x24_mi ^
--name            fold_0 ^
--display_env     lge_mi_gan_fold_0 ^
--phase           train ^
--input_nc        1 ^
--output_nc       1 ^
--nfl             64 ^
--num_downs       5 ^
--n_layers_D      2 ^
--lr              0.0002 ^
--n_epochs        125 ^
--n_epochs_decay  125 ^
--loss            CE ^
--gan             ^
--gan_mode        lsgan ^
--patch_size      96 96 24 ^
--batch_size      4 ^
--norm            instance ^
--dataaug