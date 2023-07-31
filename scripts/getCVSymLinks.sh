#!/bin/bash

python /home/maxi/Segmentation/Code/3Dseg/preprocess/CV/getCVSymlinks.py \
--cvPickleFilePath    /home/maxi/Segmentation/Data_paper2/MnM/cv_5Fold.pickle \
--rootPath            /home/maxi/Segmentation/Data_paper2/MnM \
--resPath             /home/maxi/Segmentation/Data_paper2/MnM/CV
