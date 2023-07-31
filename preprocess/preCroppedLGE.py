'''This preprocess the an LGE dataset:
This could be the Myosaiq or Emidec or both combined.
The important thing is they are initially cropped...

We here follow the naming of the myosaiq dataset for simplicity
but the output is my naming type -> sample/img.nii and sample/msk.nii.
So, the emidec cropped must follow the myosaiq naming for being input together 


Myosaiq Only--------------------------------------------------------------------------------------------------
Processing samples at D:/Data/RM/Myosaiq/database/training/all/images
There are 374 samples in total
Using th 10th percentile for axis 2 and mean for others:
 sw: 1.5625, sh: 1.5625, sd: 5.0

Min [80. 80. 24.] Median [80. 80. 24.] Max [80. 80. 24.] shapes
Min [1.5625 1.5625 5.    ] Median [1.5625 1.5625 5.    ] Max [1.5625 1.5625 5.    ] spacings
Total duration of processing: 130.8380298614502 s


Myosaiq+Emidec--------------------------------------------------------------------------------------------------
Processing samples at D:/Segmentation/Data_paper2/LGE/baseline/images
There are 440 samples in total
Using th 10th percentile for axis 2 and mean for others:
 sw: 1.5625, sh: 1.5625, sd: 5.0

Min [80. 80. 24.] Median [80. 80. 24.] Max [80. 80. 24.] shapes
Min [1.5625 1.5625 5.    ] Median [1.5625 1.5625 5.    ] Max [1.5625 1.5625 5.    ] spacings
Total duration of processing: 259.20196056365967 s

OR
This allows more complex net without modifying conv3d too much
Min [96. 96. 24.] Median [96. 96. 24.] Max [96. 96. 24.] shapes
Min [1.5625 1.5625 5.    ] Median [1.5625 1.5625 5.    ] Max [1.5625 1.5625 5.    ] spacings
'''

import os
import argparse
import matplotlib.pyplot as plt
import time
import torchio as tio
import numpy as np
import pathlib
import pickle


def proccess(dataPath, widthHeightDeep, spacings, maxLabel, resPath=None, save=False):

    samples = sorted([x for x in os.listdir(dataPath)])
    print('There are {} samples in total'.format(len(samples)))

    sw, sh, sd = np.median(spacings, 0)
    sd = np.percentile(spacings[:,2],10)
    print("Using th 10th percentile for axis 2 and mean for others:\n sw: {}, sh: {}, sd: {}".format(sw, sh, sd))
    
    newshapes    = np.zeros([len(samples),3])
    newspacings  = np.zeros([len(samples),3])
    subjectsPath = os.path.join(resPath, "subjects_preprocessed")
    plotPath     = os.path.join(resPath, "plots_preprocessed")
    if not os.path.exists(subjectsPath): pathlib.Path(subjectsPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(plotPath): pathlib.Path(plotPath).mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):

        sample = sample.split('.')[0]
        print('Processing sample = {}'.format(sample))
        newdataPath = "/".join(dataPath.split("/")[:-1])
        imgPath = os.path.join(newdataPath, "images", "{}.nii.gz".format(sample))
        mskPath = os.path.join(newdataPath, "labels", "{}.nii.gz".format(sample))

        # Get image and mask arrays
        subjectOri = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        subjectOri.check_consistent_attribute('spacing') 
        subjectOri.check_consistent_attribute('affine')
        subjectOri.check_consistent_attribute('shape')

        #Keep biggest island - no as mi and mvo might be erased
        #Resample for common spacing
        #First linear interpolation in 2D
        trans = tio.Resample((sw, sh, subjectOri.img.spacing[2]), image_interpolation='linear', label_interpolation='nearest')
        subject = trans(subjectOri)

        #And NN in out-of-plane
        trans = tio.Resample((sw, sh, sd), image_interpolation='nearest', label_interpolation='nearest')
        subject = trans(subject)

        #Padding
        refMskLabels = np.count_nonzero(subject.msk.data[0])
        shapew = widthHeightDeep[0]
        shapeh = widthHeightDeep[1]
        shaped = widthHeightDeep[2]
        trans = tio.CropOrPad((shapew, shapeh, shaped))
        subject = trans(subject)
        if not refMskLabels == np.count_nonzero(subject.msk.data[0]):
            raise ValueError("Cutting the actual mask, method is not working")

        #Normalize - not neccesary to save for invert, this should be last
        trans = tio.ZNormalization()
        subject = trans(subject)

        newshapes[i,:] = subject.img.shape[-3:]
        newspacings[i,:] = subject.img.spacing

        results = {}
        results["img_Ori"] = subjectOri.img.data.numpy()[0]
        results["msk_Ori"] = subjectOri.msk.data.numpy()[0]
        results["img"] = subject.img.data.numpy()[0]
        results["msk"] = subject.msk.data.numpy()[0]


        if np.any(results["msk"]>3) : sample = sample + "_mvo"
        else:                         sample = sample + "_mi"
        results["msk"][results["msk"]>maxLabel] = maxLabel
        subject.msk.data[subject.msk.data>maxLabel] = maxLabel

        # Save and/or plot 3D arrays
        if save:
            f, ax = plt.subplots(2,2)
            sOri = int(np.round(results["img_Ori"].shape[-1]/2))
            s = int(np.round(results["img"].shape[-1]/2))
            ax[0,0].imshow(results["img_Ori"][:,:,sOri], vmin=results["img_Ori"].min(), vmax = results["img_Ori"].max(), interpolation='none')
            ax[0,1].imshow(results["msk_Ori"][:,:,sOri], vmin=0, vmax=maxLabel, interpolation='none')
            ax[1,0].imshow(results["img"][:,:,s], vmin=results["img"].min(), vmax = results["img"].max(), interpolation='none')
            ax[1,1].imshow(results["msk"][:,:,s], vmin=0, vmax=maxLabel, interpolation='none')
            plt.savefig(os.path.join(plotPath, "{}.png".format(sample)))
            plt.close()
        
            samplePath = os.path.join(resPath, "vols_preprocessed", sample)
            if not os.path.exists(samplePath): pathlib.Path(samplePath).mkdir(parents=True, exist_ok=True)
            subject.img.save(os.path.join(samplePath, 'img.nii'), squeeze=True)
            subject.msk.save(os.path.join(samplePath, 'msk.nii'), squeeze=True)

            with open(os.path.join(subjectsPath, "{}.pickle".format(sample)), 'wb') as f:
                pickle.dump(subject, f)

    res_dict = {"voxels": newshapes, "spacings": newspacings}
    with open(os.path.join(resPath, "shapes_spacing_preprocessed.pickle"), 'wb') as f:
        pickle.dump(res_dict, f)
    
    print("Min {} Median {} Max {} shapes".format(np.min(newshapes,0), np.median(newshapes,0), np.max(newshapes,0)))
    print("Min {} Median {} Max {} spacings".format(np.min(newspacings,0), np.median(newspacings,0), np.max(newspacings,0)))

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',     type=str)
    parser.add_argument('--resPath',      type=str)
    parser.add_argument('--spacAfterCrop',type=str, required=True)
    parser.add_argument('--maxLabel',     type=int, required=True)
    parser.add_argument('--size',         type=int, nargs='+', required=True)
    args = parser.parse_args()

    start = time.time()
    with open(args.spacAfterCrop, 'rb') as handle:
        shapesspacings = pickle.load(handle)
    shapes = shapesspacings["shapes"]
    spacings = shapesspacings["spacings"]

    print("Processing samples at {}".format(args.filePath))
    proccess(args.filePath, args.size, spacings, args.maxLabel, resPath=args.resPath, save=True)
    print("Total duration of processing: {} s ".format(time.time()-start))

if __name__ == '__main__':
    main()