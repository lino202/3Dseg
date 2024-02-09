@REM @ECHO OFF
call conda activate torchTopo

@REM You can delete the segmentation folders in every sample's folder and re launch

@REM @REM Preprocessing CINE -------------------------

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/he/sample1/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/he/sample1/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/MnM/shapes_spacing_after_resampling.pickle ^
@REM --dataType      Cine

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/he/sample2/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/he/sample2/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/MnM/shapes_spacing_after_resampling.pickle ^
@REM --dataType      Cine

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/he/sample3/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/he/sample3/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/MnM/shapes_spacing_after_resampling.pickle ^
@REM --dataType      Cine


@REM @REM Preprocessing LGE -------------------------

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/cx/sample4/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/cx/sample4/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/cx/sample5/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/cx/sample5/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/cx/sample6/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/cx/sample6/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/cx/sample7/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/cx/sample7/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/cx/sampleX/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/cx/sampleX/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/la/sample8/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/la/sample8/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/la/sample9/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/la/sample9/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/la/sample11/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/la/sample11/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/preProcess.py ^
@REM --filePath      D:/Paper3/Models/invivo/mi/la/sample12/init/cropped.nii ^
@REM --resPath       D:/Paper3/Models/invivo/mi/la/sample12/segmentation ^
@REM --spacAfterCrop D:/Segmentation/Data_paper2/LGE/shapes_spacing_after_resampling.pickle ^
@REM --dataType      LGE


@REM @REM Prediction Cine -------------------------------------

@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/he/sample1/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/MnM ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           160 160 20 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_CINE


@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/he/sample2/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/MnM ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           160 160 20 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_CINE


@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/he/sample3/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/MnM ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           160 160 20 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_CINE

@REM @REM Prediction LGE -------------------------------------

@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/cx/sample4/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE


@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/cx/sample5/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/cx/sample6/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/cx/sample7/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/cx/sampleX/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/la/sample8/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE

@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/la/sample9/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE


@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/la/sample11/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE


@REM python D:/Segmentation/Code/3Dseg/newSample/pred.py ^
@REM --root_path            D:/Paper3/Models/invivo/mi/la/sample12/segmentation/for_net ^
@REM --results_dir          D:/Segmentation/Results_paper2/LGE/96x96x24_mi ^
@REM --name                 base_deform_fold ^
@REM --phase                pred ^
@REM --input_nc             1 ^
@REM --output_nc            4 ^
@REM --nfl                  64 ^
@REM --num_downs            5 ^
@REM --patch_size           96 96 24 ^
@REM --load_filename        latest_net ^
@REM --norm                 instance ^
@REM --ph                   ^
@REM --phThres              0.5 ^
@REM --phConstruction       N ^
@REM --priorName            PRIOR_LGE


@REM PosProcessing Cine ---------------------------------

python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/he/sample1/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/he/sample1/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/he/sample1/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/he/sample1/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/he/sample2/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/he/sample2/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/he/sample2/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/he/sample2/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/he/sample3/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/he/sample3/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/he/sample3/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/he/sample3/init/pred_final.nii

@REM PosProcessing LGE ---------------------------------


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/cx/sample4/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/cx/sample4/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/cx/sample4/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/cx/sample4/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/cx/sample5/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/cx/sample5/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/cx/sample5/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/cx/sample5/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/cx/sample6/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/cx/sample6/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/cx/sample6/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/cx/sample6/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/cx/sample7/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/cx/sample7/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/cx/sample7/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/cx/sample7/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/cx/sampleX/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/cx/sampleX/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/cx/sampleX/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/cx/sampleX/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/la/sample8/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/la/sample8/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/la/sample8/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/la/sample8/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/la/sample9/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/la/sample9/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/la/sample9/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/la/sample9/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/la/sample11/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/la/sample11/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/la/sample11/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/la/sample11/init/pred_final.nii


python D:/Segmentation/Code/3Dseg/newSample/posProcess.py ^
--predPath    D:/Paper3/Models/invivo/mi/la/sample12/segmentation/pred_total.nii ^
--imgOriPath  D:/Paper3/Models/invivo/mi/la/sample12/init/cropped.nii ^
--subjectPath D:/Paper3/Models/invivo/mi/la/sample12/segmentation/auxiliar/subject.pickle ^
--resPath     D:/Paper3/Models/invivo/mi/la/sample12/init/pred_final.nii