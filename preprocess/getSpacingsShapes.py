'''
Results:
MnMs:--------------------------------------------------------------
Processing samples at D:/Data/RM/MnM/OpenDataset/all
There are 320 samples in total
Original Min [196. 192.   6.], Median [256. 256.  11.], Max [548. 512.  20.] shapes
Original Min [0.68359989 0.68359971 5.        ], Median [ 1.25  1.25 10.  ], Max [ 1.82289994  1.82290006 10.        ] spacings
Max classes: [3.] count: [320]
Total duration of processing: 48.51568293571472 s
Processing samples at D:/Data/RM/MnM/OpenDataset/all
Using th 10th percentile for axis 2 and mean for others:
 sw: 1.25, sh: 1.25, sd: 8.800000190734863

Resampled Min [159. 183.   6.], Median [304. 303.  13.], Max [401. 387.  19.] shapes
Resampled Min [1.25       1.25       8.80000019], Median [1.25       1.25       8.80000019], Max [1.25       1.25       8.80000019] spacings
Total duration of processing: 694.8011491298676 s


LGE (ONLY MI):------------------------------------------------------------------------------
Processing samples at D:/Segmentation/Data_paper2/LGE/all/images
There are 315 samples in total  (WE TOOK OUT SAMPLE 244_M1 AS CROP WITH 64X64X24 FOR THE FIXED SPACING RESAMPLING WAS CUTTING THE MASK)
Original Min [44. 46.  5.], Median [72. 74. 20.], Max [109. 114.  26.] shapes
Original Min [0.9375     0.9375     3.49999571], Median [1.5625 1.5625 5.    ], Max [ 2.60416675  2.60416675 13.        ] spacings
Max classes: [3.] count: [315]
Total duration of processing: 1.6099753379821777 s
Processing samples at D:/Segmentation/Data_paper2/LGE/all/images
Using th 10th percentile for axis 2 and mean for others:
 sw: 1.5625, sh: 1.5625, sd: 5.0

Resampled Min [59. 60. 10.], Median [70. 71. 20.], Max [83. 86. 24.] shapes
Resampled Min [1.5625 1.5625 5.    ], Median [1.5625 1.5625 5.    ], Max [1.5625 1.5625 5.    ] spacings
Total duration of processing: 1.9002878665924072 s

Myosaiq:------------------------------------------------------------------------------------
Processing samples at D:/Segmentation/Data_paper2/Myosaiq/all/images
There are 358 samples in total
Original Min [44. 46.  8.], Median [70. 70. 20.], Max [109. 114.  26.] shapes
Original Min [0.9375     0.9375     3.49999571], Median [1.5625 1.5625 5.    ], Max [ 2.77777767  2.77777767 11.        ] spacings
Max classes: [2. 3. 4.] count: [  4 289  65]
Total duration of processing: 3.3184170722961426 s
Processing samples at D:/Segmentation/Data_paper2/Myosaiq/all/images
Using th 10th percentile for axis 2 and mean for others:
 sw: 1.5625, sh: 1.5625, sd: 5.0

Resampled Min [59. 60. 12.], Median [70. 71. 20.], Max [96. 95. 24.] shapes
Resampled Min [1.5625 1.5625 5.    ], Median [1.5625 1.5625 5.    ], Max [1.5625 1.5625 5.    ] spacings
Total duration of processing: 1.962785005569458 s

Emidec:--------------------------------------------------------------------------------------
Processing samples at D:/Segmentation/Data_paper2/Emidec/original_pato
There are 67 samples in total
Original Min [141. 120.   5.], Median [241. 257.   7.], Max [305. 308.  10.] shapes
Original Min [1.3671875 1.3671875 8.       ], Median [ 1.45833337  1.45833337 10.        ], Max [ 1.875       1.875      13.03999996] spacings
Max classes: [3. 4.] count: [27 40]
Total duration of processing: 2.1155927181243896 s
Processing samples at D:/Segmentation/Data_paper2/Emidec/original_pato
Using th 10th percentile for axis 2 and mean for others:
 sw: 1.4583333730697632, sh: 1.4583333730697632, sd: 10.0

Resampled Min [141. 123.   5.], Median [241. 249.   8.], Max [337. 327.  10.] shapes
Resampled Min [ 1.45833337  1.45833337 10.        ], Median [ 1.45833337  1.45833337 10.        ], Max [ 1.45833337  1.45833337 10.        ] spacings
Total duration of processing: 1.3849704265594482 s

Exvivo:----------------------------------------------------------------------------------------------
Processing samples at D:/Segmentation/Data_paper2/Exvivo/ph_checked_onlyStandfor/all match noph_checked
There are 21 samples in total
Original Min [128. 128. 128.], Median [128. 128. 128.], Max [128. 128. 128.] shapes
Original Min [1.     1.     0.9375], Median [1.     1.     0.9375], Max [1.     1.     0.9375] spacings
Max classes: [1.] count: [21]
Total duration of processing: 0.25359630584716797 s
Processing samples at D:/Segmentation/Data_paper2/Exvivo/ph_checked_onlyStandfor/all
Using th 10th percentile for axis 2 and mean for others:
 sw: 1.0, sh: 1.0, sd: 0.9374999999999999
Resampled Min [128. 128. 128.], Median [128. 128. 129.], Max [128. 128. 129.] shapes
Resampled Min [1.     1.     0.9375], Median [1.     1.     0.9375], Max [1.     1.     0.9375] spacings
Total duration of processing: 0.9412083625793457 s
'''

import os
import argparse
import time
import torchio as tio
import numpy as np
import pickle


def getOriginals(dataPath, resPath=None):

    samples = sorted([x for x in os.listdir(dataPath)])
    print('There are {} samples in total'.format(len(samples)))

    shapes = np.zeros([len(samples),3])
    spacings = np.zeros([len(samples),3])
    maxclass = np.zeros([len(samples)])

    for i, sample in enumerate(samples):
        if "MnM" in dataPath:
            imgPath = os.path.join(dataPath, sample, "{}_sa.nii.gz".format(sample))
            mskPath = os.path.join(dataPath, sample, "{}_sa_gt.nii.gz".format(sample))
        elif "Emidec" in dataPath:
            imgPath = os.path.join(dataPath, sample, "Images", "{}.nii.gz".format(sample))
            mskPath = os.path.join(dataPath, sample, "Contours", "{}.nii.gz".format(sample))
        elif "Myosaiq" in dataPath or "LGE" in dataPath:
            newdataPath = "/".join(dataPath.split("/")[:-1])
            imgPath = os.path.join(newdataPath, "images", sample)
            mskPath = os.path.join(newdataPath, "labels", sample)
        elif "Exvivo" in dataPath:
            imgPath = os.path.join(dataPath, sample,"img.nii")
            mskPath = os.path.join(dataPath, sample, "msk.nii")
        else:
            raise ValueError("Dataset not implemented")
    
        subjectOri = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        shapes[i,:] = subjectOri.shape[-3:]
        spacings[i,:] = subjectOri.spacing
        maxclass[i] = subjectOri.msk.data.numpy().max()

    res_dict = {"shapes": shapes, "spacings": spacings, "maxclass": maxclass}
    with open(os.path.join(resPath, "shapes_spacing_original.pickle"), 'wb') as f:
        pickle.dump(res_dict, f)
    
    print("Original Min {}, Median {}, Max {} shapes".format(np.min(shapes,0), np.median(shapes,0), np.max(shapes,0)))
    print("Original Min {}, Median {}, Max {} spacings".format(np.min(spacings,0), np.median(spacings,0), np.max(spacings,0)))
    classes, count = np.unique(maxclass, return_counts=True)
    print("Max classes: {} count: {}".format(classes, count))

    return shapes, spacings


def getWhenResampled(dataPath, shapes, spacings, resPath=None):

    samples = sorted([x for x in os.listdir(dataPath)])
    sw, sh, sd = np.median(spacings, 0)
    sd = np.percentile(spacings[:,2],10)
    print("Using th 10th percentile for axis 2 and mean for others:\n sw: {}, sh: {}, sd: {}".format(sw, sh, sd))

    newshapes = np.zeros([len(samples),3])
    newspacings = np.zeros([len(samples),3])

    for i, sample in enumerate(samples):

        print('preProcessing sample = {}'.format(sample))
        if "MnM" in dataPath:
            imgPath = os.path.join(dataPath, sample, "{}_sa.nii.gz".format(sample))
            mskPath = os.path.join(dataPath, sample, "{}_sa_gt.nii.gz".format(sample))
        elif "Emidec" in dataPath:
            imgPath = os.path.join(dataPath, sample, "Images", "{}.nii.gz".format(sample))
            mskPath = os.path.join(dataPath, sample, "Contours", "{}.nii.gz".format(sample))
        elif "Myosaiq" in dataPath or "LGE" in dataPath:
            newdataPath = "/".join(dataPath.split("/")[:-1])
            imgPath = os.path.join(newdataPath, "images", sample)
            mskPath = os.path.join(newdataPath, "labels", sample)
        elif "Exvivo" in dataPath:
            imgPath = os.path.join(dataPath, sample,"img.nii")
            mskPath = os.path.join(dataPath, sample, "msk.nii")
        else:
            raise ValueError("Dataset not implemented")
        subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        trans = tio.Resample((sw, sh, sd))
        subject = trans(subject)
        newshapes[i,:] = subject.img.shape[-3:]
        newspacings[i,:] = subject.img.spacing
    
    res_dict = {"shapes": newshapes, "spacings": newspacings}
    with open(os.path.join(resPath, "shapes_spacing_after_resampling.pickle"), 'wb') as f:
        pickle.dump(res_dict, f)
    
    print("Resampled Min {}, Median {}, Max {} shapes".format(np.min(newshapes,0), np.median(newshapes,0), np.max(newshapes,0)))
    print("Resampled Min {}, Median {}, Max {} spacings".format(np.min(newspacings,0), np.median(newspacings,0), np.max(newspacings,0)))

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',   type=str)
    parser.add_argument('--resPath',    type=str)
    args = parser.parse_args()
    
    start = time.time()
    print("Processing samples at {}".format(args.filePath))
    shapes, spacings = getOriginals(args.filePath, resPath=args.resPath)
    print("Total duration of processing: {} s ".format(time.time()-start))

    start = time.time()
    print("Processing samples at {}".format(args.filePath))
    getWhenResampled(args.filePath, shapes, spacings, args.resPath)
    print("Total duration of processing: {} s ".format(time.time()-start))


if __name__ == '__main__':
    main()