call conda activate torchTopo

@REM @echo off

@REM BASELINE
FOR %%i IN (0 1 2 3 4) DO (
    echo Fold %%i
	python D:/Segmentation/Code/3Dseg/pred.py ^
	--root_path            D:/Segmentation/Data_paper2/LGE/64x64x24/CV/fold_%%i/val ^
	--results_dir          D:/Segmentation/Results_paper2/LGE/64x64x24 ^
	--name                 fold_%%i ^
	--phase                pred ^
	--input_nc             1 ^
	--output_nc            4 ^
	--nfl                  64 ^
	--num_downs            5 ^
	--patch_size           64 64 24 ^
	--load_filename        latest_net ^
	--norm                 instance ^
	--phThres              0.5 ^
	--phConstruction       N ^
	--priorName            PRIOR_LGE
)

@REM TC
FOR %%i IN (0 1 2 3 4) DO (
    echo Fold %%i
	python D:/Segmentation/Code/3Dseg/pred.py ^
	--root_path            D:/Segmentation/Data_paper2/LGE/64x64x24/CV/fold_%%i/val ^
	--results_dir          D:/Segmentation/Results_paper2/LGE/64x64x24 ^
	--name                 fold_%%i ^
	--phase                pred ^
	--input_nc             1 ^
	--output_nc            4 ^
	--nfl                  64 ^
	--num_downs            5 ^
	--patch_size           64 64 24 ^
	--load_filename        latest_net ^
	--norm                 instance ^
	--ph                   ^
	--phThres              0.5 ^
	--phConstruction       N ^
	--priorName            PRIOR_LGE
)


@REM SA
FOR %%i IN (0 1 2 3 4) DO (
    echo Fold %%i
	python D:/Segmentation/Code/3Dseg/pred.py ^
	--root_path            D:/Segmentation/Data_paper2/LGE/64x64x24/CV_deform/fold_%%i/val ^
	--results_dir          D:/Segmentation/Results_paper2/LGE/64x64x24 ^
	--name                 fold_deform_%%i ^
	--phase                pred ^
	--input_nc             1 ^
	--output_nc            4 ^
	--nfl                  64 ^
	--num_downs            5 ^
	--patch_size           64 64 24 ^
	--load_filename        latest_net ^
	--norm                 instance ^
	--phThres              0.5 ^
	--phConstruction       N ^
	--priorName            PRIOR_LGE
)

@REM SATC
FOR %%i IN (0 1 2 3 4) DO (
    echo Fold %%i
	python D:/Segmentation/Code/3Dseg/pred.py ^
	--root_path            D:/Segmentation/Data_paper2/LGE/64x64x24/CV_deform/fold_%%i/val ^
	--results_dir          D:/Segmentation/Results_paper2/LGE/64x64x24 ^
	--name                 fold_deform_%%i ^
	--phase                pred ^
	--input_nc             1 ^
	--output_nc            4 ^
	--nfl                  64 ^
	--num_downs            5 ^
	--patch_size           64 64 24 ^
	--load_filename        latest_net ^
	--norm                 instance ^
	--ph                   ^
	--phThres              0.5 ^
	--phConstruction       N ^
	--priorName            PRIOR_LGE
)

