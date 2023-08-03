#!/bin/bash

python /home/maxi/Segmentation/Code/3Dseg/preprocess/CV/getCVSymlinksDeformMsk.py \
--cvPickleFilePath    /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/cv_7Fold.pickle \
--rootPath            /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/vols_preprocessed_deform \
--resPath             /home/maxi/Segmentation/Data_paper2/Exvivo/noph_checked_onlyStandfor/CV_deform


python /home/maxi/Segmentation/Code/3Dseg/preprocess/CV/getCVSymlinksDeformMsk.py \
--cvPickleFilePath    /home/maxi/Segmentation/Data_paper2/MnM/cv_5Fold.pickle \
--rootPath            /home/maxi/Segmentation/Data_paper2/MnM/vols_preprocessed_deform \
--resPath             /home/maxi/Segmentation/Data_paper2/MnM/CV_deform


#python /home/maxi/Segmentation/Code/3Dseg/preprocess/CV/getCVSymlinksDeformMsk.py \
#--cvPickleFilePath    /home/maxi/Segmentation/Data_paper2/LGE/cv_5Fold.pickle \
#--rootPath            /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/vols_preprocessed_mi_deform \
#--resPath             /home/maxi/Segmentation/Data_paper2/LGE/96x96x24/CV_mi_deform
