call conda activate torchTopo

@REM @echo off
FOR %%i IN (2) DO (
@REM FOR %%i IN (0) DO (
    echo Iteration %%i
	python D:/Segmentation/Code/3Dseg/train.py ^
	--root_path       D:/Segmentation/Data_paper2/LGE/64x64x24/CV_deform/fold_%%i ^
	--results_dir     D:/Segmentation/Results_paper2/LGE/64x64x24 ^
	--name            fold_deform_%%i ^
	--display_env     lge_mi_deform_fold_%%i ^
	--phase           train ^
	--input_nc        1 ^
	--output_nc       4 ^
	--nfl             64 ^
	--num_downs       5 ^
	--lr              0.0002 ^
	--n_epochs        125 ^
	--n_epochs_decay  125 ^
	--loss            CE ^
	--patch_size      64 64 24 ^
	--batch_size      4 ^
	--norm            instance ^
	--dataaug
)


@REM FOR %%i IN (1 2 3 4) DO (
@REM @REM FOR %%i IN (0) DO (
@REM     echo Iteration %%i
@REM 	python D:/Segmentation/Code/3Dseg/trainGAN.py ^
@REM 	--root_path       D:/Segmentation/Data_paper2/LGE/64x64x24/CV/fold_%%i ^
@REM 	--results_dir     D:/Segmentation/Results_paper2/LGE_GAN/64x64x24 ^
@REM 	--name            fold_%%i ^
@REM 	--display_env     lge_mi_gan_fold_%%i ^
@REM 	--phase           train ^
@REM 	--input_nc        1 ^
@REM 	--output_nc       1 ^
@REM 	--nfl             64 ^
@REM 	--num_downs       5 ^
@REM 	--n_layers_D      2 ^
@REM 	--lr              0.0002 ^
@REM 	--n_epochs        125 ^
@REM 	--n_epochs_decay  125 ^
@REM 	--loss            CE ^
@REM 	--gan             ^
@REM 	--gan_mode        lsgan ^
@REM 	--patch_size      64 64 24 ^
@REM 	--batch_size      4 ^
@REM 	--norm            instance ^
@REM 	--dataaug
@REM )

