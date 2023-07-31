'''This preprocess the Emidec dataset:
Initially neither slicer neither itk-snap read the files correctly
Also only the training data gt is available
Also we cannot crop Roi in here with cine but the LV is centered so 
we crop in the xy axis.

As this is used with Myosaiq we output the cropped volumes with that dataset naming

Results:
Spacing in mm
Min [64. 64.  5.] Median [83. 83.  7.] Max [88. 88. 10.] shapes
Min [1.3671875 1.3671875 8.       ] Median [ 1.45833337  1.45833337 10.        ] Max [ 1.875       1.875      13.03999996] spacings
'''

import os
import argparse
import torchio as tio
import numpy as np
import pathlib
import pickle
import matplotlib.pyplot as plt


def proccess(dataPath, space, resPath=None, save=False):

    samples = sorted([x for x in os.listdir(dataPath)])
    print('There are {} samples in total'.format(len(samples)))

    newshapes    = np.zeros([len(samples),3])
    newspacings  = np.zeros([len(samples),3])
    subjectsPath = os.path.join(resPath, "subjects_preprocessed")
    plotPath     = os.path.join(resPath, "plots_preprocessed")
    imgResPath   = os.path.join(resPath, "vols_preprocessed", "images")
    mskResPath   = os.path.join(resPath, "vols_preprocessed", "labels")
    if not os.path.exists(subjectsPath): pathlib.Path(subjectsPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(plotPath): pathlib.Path(plotPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(imgResPath): pathlib.Path(imgResPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(mskResPath): pathlib.Path(mskResPath).mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):

        print('Processing sample = {}'.format(sample))
        imgPath = os.path.join(dataPath, sample, "Images", "{}.nii.gz".format(sample))
        mskPath = os.path.join(dataPath, sample, "Contours",   "{}.nii.gz".format(sample))
        
        # Get image and mask arrays
        subjectOri = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        subjectOri.check_consistent_attribute('spacing') 
        subjectOri.check_consistent_attribute('affine')
        subjectOri.check_consistent_attribute('shape')

        #Crop
        refMskLabels = np.count_nonzero(subjectOri.msk.data[0])
        shapew = int(np.ceil(space[0]/subjectOri.spacing[0]))
        shapeh = int(np.ceil(space[1]/subjectOri.spacing[1]))
        shaped = subjectOri.msk.shape[3]
        trans = tio.CropOrPad((shapew, shapeh, shaped))
        subject = trans(subjectOri)
        if not refMskLabels == np.count_nonzero(subject.msk.data[0]):
            raise ValueError("Cutting the actual mask, method is not working")

        newshapes[i,:]   = subject.img.shape[-3:]
        newspacings[i,:] = subject.img.spacing

        results = {}
        results["img_Ori"] = subjectOri.img.data.numpy()[0]
        results["msk_Ori"] = subjectOri.msk.data.numpy()[0]
        results["img"] = subject.img.data.numpy()[0]
        results["msk"] = subject.msk.data.numpy()[0]

        # Save and/or plot 3D arrays  
        if save:
            f, ax = plt.subplots(2,2)
            sOri = int(np.round(results["img_Ori"].shape[-1]/2))
            s = int(np.round(results["img"].shape[-1]/2))
            ax[0,0].imshow(results["img_Ori"][:,:,sOri], vmin=results["img_Ori"].min(), vmax = results["img_Ori"].max(), interpolation='none')
            ax[0,1].imshow(results["msk_Ori"][:,:,sOri], vmin=0, vmax=4, interpolation='none')
            ax[1,0].imshow(results["img"][:,:,s], vmin=results["img"].min(), vmax = results["img"].max(), interpolation='none')
            ax[1,1].imshow(results["msk"][:,:,s], vmin=0, vmax=4, interpolation='none')
            plt.savefig(os.path.join(plotPath, "{}.png".format(sample)))
            plt.close()

            subject.img.save(os.path.join(imgResPath, '{}.nii.gz'.format(sample)), squeeze=True)
            subject.msk.save(os.path.join(mskResPath, '{}.nii.gz'.format(sample)), squeeze=True)

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
    args = parser.parse_args()

    space = [120, 120] # space in SA plane of 120x120mm as LV is centered 
    proccess(args.filePath, space, resPath=args.resPath, save=True)

if __name__ == '__main__':
    main()