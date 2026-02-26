"""Here we transform the volume to have the original shape and spacing
respecting the initial spacial orientation. Here we only work the original msk
and the prediction. In computeMetrics we get plots and everything for the original 
img, msks and preds or for the preds without this transformation to the original space
"""

import os
import argparse
import numpy as np
import pickle
import time
import pathlib
import torchio as tio
import sys


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--msksFolder',   type=str, required=True)
    parser.add_argument('--predsFolder',  type=str, required=True)
    parser.add_argument('--subPreFolder', type=str, required=True)
    parser.add_argument('--resPath',      type=str)
    parser.add_argument('--rootCodePath', type=str)
    args = parser.parse_args()

    sys.path.append(args.rootCodePath)   # This is ugly, I know
    from preprocess.utilsPre import getEXFromMask

    if "nnUNet" in args.predsFolder:
        samples = sorted([x.split(".nii.gz")[0] for x in os.listdir(args.predsFolder) if ".nii.gz" in x])
    else:
        samples = sorted([x for x in os.listdir(args.predsFolder)])
    print('There are {} samples in total'.format(len(samples)))

    for i, sample in enumerate(samples):

        print('Processing sample = {}'.format(sample))

        # Get pred and mask volumes
        if "nnUNet" in args.predsFolder:
            predPath = os.path.join(args.predsFolder, "{}.nii.gz".format(sample))
        else:
            predPath = os.path.join(args.predsFolder, sample, "pred.nii")
        pred = tio.LabelMap(predPath)
        
        if "MnM" in args.msksFolder:
            #Here we need to get the ED, (shape and spacing should be the same in 4th dim but we still search for the ED)
            mskPath = os.path.join(args.msksFolder, sample, "{}_sa_gt.nii.gz".format(sample))
            msk     = tio.LabelMap(mskPath)
            edes    = getEXFromMask(msk.data.numpy(), sample)
            msk.set_data(msk.data[edes[0]][np.newaxis, :])
        
        elif "Myosaiq" in args.msksFolder or "LGE" in args.msksFolder:
            #Here we need to delete the _mi/_mvo as the original name does not have it
            newdataPath    = "/".join(args.msksFolder.split("/")[:-1])
            origSampleName = "_".join(sample.split('_')[:-1])
            mskPath        = os.path.join(newdataPath, "labels", "{}.nii.gz".format(origSampleName))
            msk            = tio.LabelMap(mskPath)

        elif "Exvivo" in args.msksFolder:
            raise ValueError("This dataset is already in the correct spatial configuration")
            # mskPath = os.path.join(args.msksFolder, sample, "msk.nii")
            # msk  = tio.LabelMap(mskPath)

        elif "Emidec" in args.msksFolder:
            raise ValueError("The Emidec Dataset is not use directly")
            # mskPath = os.path.join(args.msksFolder, sample, "Contours", "{}.nii.gz".format(sample))

        else:
            raise ValueError("Dataset not implemented")

        # Now we get the subjects-preprocessed to invert crop and padding on prediction 
        subPrePath = os.path.join(args.subPreFolder, "{}.pickle".format(sample))
        with open(subPrePath, 'rb') as f: preprocessedSubject = pickle.load(f)

        print("Initial shape and spacing")
        print("Msk {}\nPred {}".format(msk, pred))

        #Transform
        #Padding and crop
        inverse_transform = preprocessedSubject.get_inverse_transform(warn=False)
        pred = inverse_transform(pred)

        #Final resampling, this adds uncertainty
        trans = tio.Resample(msk)
        pred = trans(pred)

        subject = tio.Subject(msk=msk, pred=pred)
        trans   = tio.CopyAffine('msk')
        subject = trans(subject)
        subject.check_consistent_attribute('spacing') 
        subject.check_consistent_attribute('affine')
        subject.check_consistent_attribute('shape')

        print("Final shape and spacing")
        print("Msk {}\nPred {}".format(msk, pred))

        #Save
        resSamplePath = os.path.join(args.resPath, sample)
        if not os.path.exists(resSamplePath): pathlib.Path(resSamplePath).mkdir(parents=True, exist_ok=True)
        subject.pred.save(os.path.join(resSamplePath, 'pred.nii'), squeeze=True)

if __name__ == '__main__':
    start = time.time()
    main()
    print("Total duration of processing: {} s ".format(time.time()-start))

