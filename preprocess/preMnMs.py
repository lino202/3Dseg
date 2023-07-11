'''This preprocess the MnMs dataset:
Slicer does not read well 4D nifti so if you loaded in python nibabel or torchio and save the 3D of ED or ES
you will see the array correctly. The location in space xyz of the volume changes. 
The affine matrix or spatial location are respected with this code as torchio modifies the affine in consequence
to the transform applied
SimpleItk must be 2.0.2 otherwise an error on some images happens ITK ERROR: ITK only supports orthonormal direction cosines. 
No orthonormal definition found!
    
Some samples get a wrong ROI due to I think the esofagus movement and brightness,
then the center is applied there and heart is cropped, so for those cases by 
use a more aggressive initial crop (see except)
sample G4I7V2 had a separate island originally

Results images with spacings in mm3
Min [160. 160.  20.] Median [160. 160.  20.] Max [160. 160.  20.] shapes
Min [1.25       1.25       8.80000019] Median [1.25       1.25       8.80000019] Max [1.25       1.25       8.80000019] spacings
with padding to achieve this. initial crop of 20 band was applied. for the ones problematic we used a higher 
initial crop of almost a half and we solve other problems by keeping biggest island as one sample had 
an isolated RV island error.
'''

import os
import argparse
import time
import torchio as tio
import numpy as np
from utilsPre import getEXFromMask, cropSubjectCINE
import pickle
import pathlib
import traceback
import matplotlib.pyplot as plt
import torch.nn.functional as F

def proccess(dataPath, initCrop, widthHeightDeep, spacings, resPath=None, save=False):

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

        print('Processing sample = {}'.format(sample))
        imgPath = os.path.join(dataPath, sample, "{}_sa.nii.gz".format(sample))
        mskPath = os.path.join(dataPath, sample, "{}_sa_gt.nii.gz".format(sample))

        # Get image and mask arrays
        subjectOri = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        subjectOri.check_consistent_attribute('spacing') 
        subjectOri.check_consistent_attribute('affine')
        subjectOri.check_consistent_attribute('shape')
        edes = getEXFromMask(subjectOri.msk.data.numpy(), sample)

        #Keep biggest island
        trans = tio.KeepLargestComponent()
        tmpED = subjectOri.msk.data[edes[0],:,:,:] # we are only interest in ED
        tmpED = F.one_hot(tmpED.long(), num_classes=int(subjectOri.msk.data.max())+1)
        tmpED = tmpED.permute(3, 0, 1, 2).type(tmpED.type())
        for fg in range(1,tmpED.shape[0]):
            subjectMsk = tio.Subject(msk=tio.LabelMap(tensor=tmpED[fg][np.newaxis,:]))
            subjectMsk = trans(subjectMsk)
            tmpED[fg] = subjectMsk.msk.data
        tmpED = tmpED.argmax(dim=0)
        subjectOri.msk.data[edes[0]] = tmpED

        #Resample for common spacing
        #First linear interpolation in 2D
        trans = tio.Resample((sw, sh, subjectOri.img.spacing[2]), image_interpolation='linear', label_interpolation='nearest')
        subject = trans(subjectOri)

        #And NN in out-of-plane
        trans = tio.Resample((sw, sh, sd), image_interpolation='nearest', label_interpolation='nearest')
        subject = trans(subject)

        roiPlotPath = os.path.join(plotPath, "{}_1HarmGauss.png".format(sample))
        #Initial Crop
        shapew = subject.img.shape[1]-int(subject.img.shape[1]/initCrop) 
        shapeh = subject.img.shape[2]-int(subject.img.shape[2]/initCrop)
        trans = tio.CropOrPad((shapew, shapeh, subject.img.shape[3]))
        subject = trans(subject)
        
        try:
            # ROI Crop both 4D arrays
            subject = cropSubjectCINE(subject, edes, widthHeightDeep[:2], roiPlotPath)
        except:
            print("Sample {} finished with exception {}".format(sample, traceback.format_exc()))
            print("Trying bigger second crop of the borders")

            #Initial Crop
            shapew = subject.img.shape[1]-int(subject.img.shape[1]/2) 
            shapeh = subject.img.shape[2]-int(subject.img.shape[2]/2)
            trans = tio.CropOrPad((shapew, shapeh, subject.img.shape[3]))
            subject = trans(subject)

            # ROI Crop both 4D arrays
            subject = cropSubjectCINE(subject, edes, widthHeightDeep[:2], roiPlotPath)

        #We get only ED to speed up last operations
        subject.img.set_data(subject.img.data[edes[0]][np.newaxis, :])
        subject.msk.set_data(subject.msk.data[edes[0]][np.newaxis, :])

        #Padding 
        shapew = widthHeightDeep[0] + widthHeightDeep[1]
        shapeh = widthHeightDeep[0] + widthHeightDeep[1]
        shaped = widthHeightDeep[2]
        trans = tio.CropOrPad((shapew, shapeh, shaped))
        subject = trans(subject)

        #Normalize - not neccesary to save for invert, this should be last
        trans = tio.ZNormalization()
        subject = trans(subject)

        newshapes[i,:] = subject.img.shape[-3:]
        newspacings[i,:] = subject.img.spacing

        #Separate ED and ES 3D arrays
        results = {}
        results["img_Ori"] = subjectOri.img.data.numpy()[edes[0]]
        results["msk_Ori"] = subjectOri.msk.data.numpy()[edes[0]]
        results["img"] = subject.img.data.numpy()[0]
        results["msk"] = subject.msk.data.numpy()[0]

        # Save and/or plot 3D ED and ES arrays        
        if save:
            f, ax = plt.subplots(2,2)
            sOri = int(np.round(results["img_Ori"].shape[-1]/2))
            s = int(np.round(results["img"].shape[-1]/2))
            ax[0,0].imshow(results["img_Ori"][:,:,sOri], vmin=results["img_Ori"].min(), vmax = results["img_Ori"].max())
            ax[0,1].imshow(results["msk_Ori"][:,:,sOri], vmin=0, vmax=3)
            ax[1,0].imshow(results["img"][:,:,s], vmin=results["img"].min(), vmax = results["img"].max())
            ax[1,1].imshow(results["msk"][:,:,s], vmin=0, vmax=3)
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
    parser.add_argument('--filePath',   type=str)
    parser.add_argument('--resPath',    type=str)
    parser.add_argument('--initCrop',   type=float, default=5)
    parser.add_argument('--spacAfterCrop',type=str, required=True)
    args = parser.parse_args()

    start = time.time()
    with open(args.spacAfterCrop, 'rb') as handle:
        shapesspacings = pickle.load(handle)
    shapes = shapesspacings["shapes"]
    spacings = shapesspacings["spacings"]

    # we fix the SA to 160*160 but we need to know the voxel shape of LA for padding default is 20
    widthHeightDeep = [80,80, np.max(shapes,0)[2] + 1] # +1 in order to finish with shapes 160x160x20 
    print("Processing samples at {}".format(args.filePath))
    proccess(args.filePath, args.initCrop, widthHeightDeep, spacings, resPath=args.resPath, save=True)
    print("Total duration of processing: {} s ".format(time.time()-start))


if __name__ == '__main__':
    main()