call conda activate torchTopo

@echo off

@REM python D:/Segmentation/Code/3Dseg/postprocess/computeMetrics.py ^
@REM --predsFolder           D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_SATC_originalSpace ^
@REM --msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
@REM --imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
@REM --nClasses              4 ^
@REM --priorName             PRIOR_LGE ^
@REM --phThres               0.5 ^
@REM --resPath               D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_SATC_originalSpace_metrics ^
@REM --res_excel_indexs      gdsc_bg gdsc_lv gdsc_myo gdsc_mi hd_bg hd_lv hd_myo hd_mi assd_bg assd_lv assd_myo assd_mi be ts ^
@REM --rootCodePath          D:/Segmentation/Code/3DSeg


@REM compute separate BE for LV, MYO, MI, LV+MYO, LV+MI, MYO+MI

python D:/Segmentation/Code/3Dseg/postprocess/computeSeparateBE.py ^
--predsFolder           D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_B_originalSpace ^
--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
--nClasses              4 ^
--priorName             PRIOR_LGE ^
--phThres               0.5 ^
--resPath               D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_B_originalSpace_metrics ^
--res_excel_indexs      be_lv be_myo be_mi be_lvmyo be_lvmi be_myomi ^
--rootCodePath          D:/Segmentation/Code/3DSeg

python D:/Segmentation/Code/3Dseg/postprocess/computeSeparateBE.py ^
--predsFolder           D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_SA_originalSpace ^
--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
--nClasses              4 ^
--priorName             PRIOR_LGE ^
--phThres               0.5 ^
--resPath               D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_SA_originalSpace_metrics ^
--res_excel_indexs      be_lv be_myo be_mi be_lvmyo be_lvmi be_myomi ^
--rootCodePath          D:/Segmentation/Code/3DSeg

python D:/Segmentation/Code/3Dseg/postprocess/computeSeparateBE.py ^
--predsFolder           D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_TC_originalSpace ^
--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
--nClasses              4 ^
--priorName             PRIOR_LGE ^
--phThres               0.5 ^
--resPath               D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_TC_originalSpace_metrics ^
--res_excel_indexs      be_lv be_myo be_mi be_lvmyo be_lvmi be_myomi ^
--rootCodePath          D:/Segmentation/Code/3DSeg

python D:/Segmentation/Code/3Dseg/postprocess/computeSeparateBE.py ^
--predsFolder           D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_SATC_originalSpace ^
--msksFolder            D:/Segmentation/Data_paper2/LGE/all/labels ^
--imgsFolder            D:/Segmentation/Data_paper2/LGE/all/images ^
--nClasses              4 ^
--priorName             PRIOR_LGE ^
--phThres               0.5 ^
--resPath               D:/Segmentation/Review_paper2/nnUNet/nnUNet_results/results_SATC_originalSpace_metrics	^
--res_excel_indexs      be_lv be_myo be_mi be_lvmyo be_lvmi be_myomi ^
--rootCodePath          D:/Segmentation/Code/3DSeg

