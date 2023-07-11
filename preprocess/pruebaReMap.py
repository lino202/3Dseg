import os
import argparse
import matplotlib.pyplot as plt
import time
import torchio as tio
import numpy as np
import pathlib
import pickle


def proccess(filePath, resPath=None):

    dataPath = os.path.join(filePath, "vols_preprocessed")
    samples = sorted([x for x in os.listdir(dataPath)])
    print('There are {} samples in total'.format(len(samples)))
    transPath = os.path.join(filePath, "subjects_preprocessed")
    if not os.path.exists(resPath): pathlib.Path(resPath).mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):

        # Get image and mask arrays
        print('Processing sample = {}'.format(sample))
        mskPath = os.path.join(dataPath, sample, "msk.nii")
        mskPreprocessed = tio.LabelMap(mskPath)

        if "P" in sample:
            oriName = "_".join(sample.split("_")[:-1])
        else: oriName = sample
        mskPath = os.path.join(filePath, "original", oriName, "Contours", "{}.nii.gz".format(oriName))
        mskOri  = tio.LabelMap(mskPath)
        mskOri.data[mskOri.data>3] = 3

        transSamplePath = os.path.join(transPath, "{}.pickle".format(sample))
        with open(transSamplePath, 'rb') as f:
            preprocessedSubject = pickle.load(f)
        
        # inverse_transforms = []
        # for key in list(transforms.keys())[::-1]: #invert list of applied shape and spacing transforms
        #     if "pad" in key or "crop" in key:
        #         trans = tio.CropOrPad(transforms[key]['in_shape'][-3:])
        #         inverse_transforms.append(trans)
        #     # elif "resample" in key:
        #     #     trans = tio.Resample(transforms[key]['in_spacing'])
        #     # else: raise ValueError("Wrong transformation")
        #     # inverse_transforms.append(trans)

        # inverseTrans = tio.Compose(inverse_transforms)
        # mskReconst = inverseTrans(mskPreprocessed)
        # mskReconst = tio.LabelMap(tensor=mskReconst.data, affine=mskOri.affine)

        inverse_transform = preprocessedSubject.get_inverse_transform(warn=False)
        mskReconst = inverse_transform(mskPreprocessed)
        trans = tio.Resample(mskOri)
        mskReconst = trans(mskReconst)
        
        # print(mskOri.shape, mskReconst.shape)
        # print(mskOri.affine == mskReconst.affine)
        # print(mskOri.spacing, mskReconst.spacing)


        if np.any(mskOri.data.numpy() != mskReconst.data.numpy()): print("!!!!!Wrong not equal labels")#print("check sample {}".format(sample))
        if mskOri.shape != mskReconst.shape:                       raise ValueError("Wrong shape")#print("Wrong shape")
        if mskOri.spacing != mskReconst.spacing:                   raise ValueError("Wrong spacing")#print("Wrong spacing")
        if np.any(mskOri.affine != mskReconst.affine):             raise ValueError("Wrong affine")#print("Wrong affine")

        samplePath = os.path.join(resPath, sample)
        if not os.path.exists(samplePath): pathlib.Path(samplePath).mkdir(parents=True, exist_ok=True)
        mskReconst.save(os.path.join(samplePath, 'msk.nii'), squeeze=True)

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',   type=str)
    parser.add_argument('--resPath',    type=str)
    args = parser.parse_args()

    proccess(args.filePath, resPath=args.resPath)


if __name__ == '__main__':
    main()