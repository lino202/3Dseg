call conda activate torchTopo

@echo off

@REM @REM Return to original shape and spacing --------------------------------
@REM FOR %%i IN (0 1 2 3 4) DO (
@REM     echo Fold %%i
@REM 	python D:/Segmentation/Code/3Dseg/postprocess/getOriginalSpace.py ^
@REM 	--msksFolder    D:/Segmentation/Data_paper2/LGE/all/labels ^
@REM 	--predsFolder   D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/volumes_baseline_phconstN ^
@REM 	--subPreFolder  D:/Segmentation/Data_paper2/LGE/64x64x24/subjects_preprocessed ^
@REM 	--resPath       D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/volumes_baseline_phconstN_originalSpace ^
@REM 	--rootCodePath  D:/Segmentation/Code/3DSeg
@REM )

@REM FOR %%i IN (0 1 2 3 4) DO (
@REM     echo Fold %%i
@REM 	python D:/Segmentation/Code/3Dseg/postprocess/getOriginalSpace.py ^
@REM 	--msksFolder    D:/Segmentation/Data_paper2/LGE/all/labels ^
@REM 	--predsFolder   D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/volumes_ph_phconstN ^
@REM 	--subPreFolder  D:/Segmentation/Data_paper2/LGE/64x64x24/subjects_preprocessed ^
@REM 	--resPath       D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/volumes_ph_phconstN_originalSpace ^
@REM 	--rootCodePath  D:/Segmentation/Code/3DSeg
@REM )

@REM FOR %%i IN (0 1 2 3 4) DO (
@REM     echo Fold %%i
@REM 	python D:/Segmentation/Code/3Dseg/postprocess/getOriginalSpace.py ^
@REM 	--msksFolder    D:/Segmentation/Data_paper2/LGE/all/labels ^
@REM 	--predsFolder   D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/volumes_baseline_phconstN ^
@REM 	--subPreFolder  D:/Segmentation/Data_paper2/LGE/64x64x24/subjects_preprocessed ^
@REM 	--resPath       D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/volumes_baseline_phconstN_originalSpace ^
@REM 	--rootCodePath  D:/Segmentation/Code/3DSeg
@REM )

@REM FOR %%i IN (0 1 2 3 4) DO (
@REM     echo Fold %%i
@REM 	python D:/Segmentation/Code/3Dseg/postprocess/getOriginalSpace.py ^
@REM 	--msksFolder    D:/Segmentation/Data_paper2/LGE/all/labels ^
@REM 	--predsFolder   D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/volumes_ph_phconstN ^
@REM 	--subPreFolder  D:/Segmentation/Data_paper2/LGE/64x64x24/subjects_preprocessed ^
@REM 	--resPath       D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/volumes_ph_phconstN_originalSpace ^
@REM 	--rootCodePath  D:/Segmentation/Code/3DSeg
@REM )



@REM compute metrics --------------------------------------------------------------------------------
FOR %%i IN (0 1 2 3 4) DO (
    echo Fold %%i
	python D:/Segmentation/Code/3Dseg/postprocess/computeMetrics.py ^
	--predsFolder           D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/volumes_baseline_phconstN_originalSpace ^
	--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
	--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
	--nClasses              4 ^
	--priorName             PRIOR_LGE ^
	--phThres               0.5 ^
	--resPath               D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/metrics/volumes_baseline_phconstN_originalSpace ^
	--res_excel_indexs      gdsc_bg gdsc_lv gdsc_myo gdsc_mi hd_bg hd_lv hd_myo hd_mi assd_bg assd_lv assd_myo assd_mi be ts ^
	--rootCodePath          D:/Segmentation/Code/3DSeg

	python D:/Segmentation/Code/3Dseg/postprocess/computeSeparateBE.py ^
	--predsFolder           D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/volumes_baseline_phconstN_originalSpace ^
	--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
	--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
	--nClasses              4 ^
	--priorName             PRIOR_LGE ^
	--phThres               0.5 ^
	--resPath               D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/metrics/volumes_baseline_phconstN_originalSpace ^
	--res_excel_indexs      be_lv be_myo be_mi be_lvmyo be_lvmi be_myomi ^
	--rootCodePath          D:/Segmentation/Code/3DSeg
)


FOR %%i IN (0 1 2 3 4) DO (
    echo Fold %%i
	python D:/Segmentation/Code/3Dseg/postprocess/computeMetrics.py ^
	--predsFolder           D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/volumes_ph_phconstN_originalSpace ^
	--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
	--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
	--nClasses              4 ^
	--priorName             PRIOR_LGE ^
	--phThres               0.5 ^
	--resPath               D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/metrics/volumes_ph_phconstN_originalSpace ^
	--res_excel_indexs      gdsc_bg gdsc_lv gdsc_myo gdsc_mi hd_bg hd_lv hd_myo hd_mi assd_bg assd_lv assd_myo assd_mi be ts ^
	--rootCodePath          D:/Segmentation/Code/3DSeg

	python D:/Segmentation/Code/3Dseg/postprocess/computeSeparateBE.py ^
	--predsFolder           D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/volumes_ph_phconstN_originalSpace ^
	--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
	--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
	--nClasses              4 ^
	--priorName             PRIOR_LGE ^
	--phThres               0.5 ^
	--resPath               D:/Segmentation/Results_paper2/LGE/64x64x24/fold_%%i/metrics/volumes_ph_phconstN_originalSpace ^
	--res_excel_indexs      be_lv be_myo be_mi be_lvmyo be_lvmi be_myomi ^
	--rootCodePath          D:/Segmentation/Code/3DSeg
)


FOR %%i IN (0 1 2 3 4) DO (
    echo Fold %%i
	python D:/Segmentation/Code/3Dseg/postprocess/computeMetrics.py ^
	--predsFolder           D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/volumes_baseline_phconstN_originalSpace ^
	--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
	--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
	--nClasses              4 ^
	--priorName             PRIOR_LGE ^
	--phThres               0.5 ^
	--resPath               D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/metrics/volumes_baseline_phconstN_originalSpace ^
	--res_excel_indexs      gdsc_bg gdsc_lv gdsc_myo gdsc_mi hd_bg hd_lv hd_myo hd_mi assd_bg assd_lv assd_myo assd_mi be ts ^
	--rootCodePath          D:/Segmentation/Code/3DSeg

	python D:/Segmentation/Code/3Dseg/postprocess/computeSeparateBE.py ^
	--predsFolder           D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/volumes_baseline_phconstN_originalSpace ^
	--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
	--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
	--nClasses              4 ^
	--priorName             PRIOR_LGE ^
	--phThres               0.5 ^
	--resPath               D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/metrics/volumes_baseline_phconstN_originalSpace ^
	--res_excel_indexs      be_lv be_myo be_mi be_lvmyo be_lvmi be_myomi ^
	--rootCodePath          D:/Segmentation/Code/3DSeg
)


FOR %%i IN (0 1 2 3 4) DO (
    echo Fold %%i
	python D:/Segmentation/Code/3Dseg/postprocess/computeMetrics.py ^
	--predsFolder           D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/volumes_ph_phconstN_originalSpace ^
	--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
	--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
	--nClasses              4 ^
	--priorName             PRIOR_LGE ^
	--phThres               0.5 ^
	--resPath               D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/metrics/volumes_ph_phconstN_originalSpace ^
	--res_excel_indexs      gdsc_bg gdsc_lv gdsc_myo gdsc_mi hd_bg hd_lv hd_myo hd_mi assd_bg assd_lv assd_myo assd_mi be ts ^
	--rootCodePath          D:/Segmentation/Code/3DSeg

	python D:/Segmentation/Code/3Dseg/postprocess/computeSeparateBE.py ^
	--predsFolder           D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/volumes_ph_phconstN_originalSpace ^
	--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
	--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
	--nClasses              4 ^
	--priorName             PRIOR_LGE ^
	--phThres               0.5 ^
	--resPath               D:/Segmentation/Results_paper2/LGE/64x64x24/fold_deform_%%i/metrics/volumes_ph_phconstN_originalSpace ^
	--res_excel_indexs      be_lv be_myo be_mi be_lvmyo be_lvmi be_myomi ^
	--rootCodePath          D:/Segmentation/Code/3DSeg
)



