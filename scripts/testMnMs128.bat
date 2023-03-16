@REM @ECHO OFF
call conda activate torchTopo

python F:/Segmentation/Code/3Dseg/test.py ^
--root_path        F:/Segmentation/Data/seg3D_MnMs/128x128 ^
--results_dir      F:/Segmentation/Results/seg3D_MnMs/128x128 ^
--name             init ^
--phase            test ^
--input_nc         1 ^
--output_nc        4 ^
--nfl              64 ^
--num_downs        7 ^
--load_size_h      128 ^
--load_size_w      128 ^
--load_size_d      128 ^
--load_filename    latest_net ^
--res_excel        F:/Segmentation/Results/seg3D_MnMs/128x128/results2.xlsx ^
--res_params_name  results_baseline ^
--res_excel_indexs gdsc_bg gdsc_lv gdsc_myo gdsc_rv hd_bg hd_lv hd_myo hd_rv be ts