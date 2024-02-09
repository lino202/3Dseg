'''This preprocess the Cine ED, LGE, Exvivo images.

We assume the cine already is the ED and also all samples are already cropped!!

We need to have sizes smaller than:
160x160x20  in Cine
96x96x24    in LGE
128x128x128 in Exvivo, here the sample is isotropic in spacing 1x1x0.9375 and with a shape of 128x128x128 
            so the only preprocessing is Znormalization, if the volume is not in this state you should probably make it
            in that shape and spacing and save the trnasformation for posterior reshaping and respacing in the original
            space, BUT we assume this is not needed as the exvivo dataset used here was really standard and simple
            and new data might not even give good results but hey! you could try!
'''

import os
import argparse
import time
import torchio as tio
import numpy as np
import pickle
import pathlib
import matplotlib.pyplot as plt

def preproccessCineLGE(imgPath, widthHeightDeep, spacAfterCrop, resPath=None):

    with open(spacAfterCrop, 'rb') as handle:
        shapesspacings = pickle.load(handle)
    spacings = shapesspacings["spacings"]

    sw, sh, sd = np.median(spacings, 0)
    sd = np.percentile(spacings[:,2],10)
    print("Using th 10th percentile for axis 2 and mean for others:\n sw: {}, sh: {}, sd: {}".format(sw, sh, sd))
    auxiliarPath = os.path.join(resPath, 'auxiliar')
    forNetPath   = os.path.join(resPath, 'for_net')
    if not os.path.exists(resPath): pathlib.Path(resPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(auxiliarPath): pathlib.Path(auxiliarPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(forNetPath): pathlib.Path(forNetPath).mkdir(parents=True, exist_ok=True)

    # Get image 
    print('Processing sample = {}'.format(imgPath))
    subjectOri = tio.Subject(img=tio.ScalarImage(imgPath))
    print("Original shape: {}".format(subjectOri.img.shape))
    print("Original spacing: {}".format(subjectOri.img.spacing))

    #Resample for common spacing
    #First linear interpolation in 2D
    trans = tio.Resample((sw, sh, subjectOri.img.spacing[2]), image_interpolation='linear')
    subject = trans(subjectOri)

    #And NN in out-of-plane
    trans = tio.Resample((sw, sh, sd), image_interpolation='nearest')
    subject = trans(subject)

    #Padding 
    shapew = widthHeightDeep[0]
    shapeh = widthHeightDeep[1]
    shaped = widthHeightDeep[2]
    trans = tio.CropOrPad((shapew, shapeh, shaped))
    subject = trans(subject)

    #Normalize - not neccesary to save for invert, this should be last
    trans = tio.ZNormalization()
    subject = trans(subject)

    imgArr = subject.img.data.numpy()
    plt.figure()
    s = int(np.round(imgArr.shape[-1]/2))
    plt.imshow(imgArr[0,:,:,s], vmin=imgArr.min(), vmax = imgArr.max(), interpolation='none')
    plt.savefig(os.path.join(auxiliarPath, "check_SAX.png"))
    plt.close()

    samplePath = os.path.join(forNetPath, 'pred')
    if not os.path.exists(samplePath): pathlib.Path(samplePath).mkdir(parents=True, exist_ok=True)
    subject.img.save(os.path.join(samplePath, 'img.nii'), squeeze=True)

    with open(os.path.join(auxiliarPath, "subject.pickle"), 'wb') as f:
        pickle.dump(subject, f)
    
    print("New shape: {}".format(subject.img.shape))
    print("New spacing: {}".format(subject.img.spacing))
    

def preproccessExvivo(imgPath, resPath=None):

    auxiliarPath = os.path.join(resPath, 'auxiliar')
    forNetPath   = os.path.join(resPath, 'for_net')
    if not os.path.exists(resPath): pathlib.Path(resPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(auxiliarPath): pathlib.Path(auxiliarPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(forNetPath): pathlib.Path(forNetPath).mkdir(parents=True, exist_ok=True)

    # Get image 
    print('Processing sample = {}'.format(imgPath))
    subjectOri = tio.Subject(img=tio.ScalarImage(imgPath))
    print("Original shape: {}".format(subjectOri.img.shape))
    print("Original spacing: {}".format(subjectOri.img.spacing))

    #Resample and crop/padding is not neccesary here as samples are 128x128x128 shape and 1x1x0.9375 spacing

    #Normalize - not neccesary to save for invert, this should be last
    trans = tio.ZNormalization()
    subject = trans(subjectOri)

    imgArr = subject.img.data.numpy()
    plt.figure()
    s = int(np.round(imgArr.shape[-1]/2))
    plt.imshow(imgArr[0,:,:,s], vmin=imgArr.min(), vmax = imgArr.max(), interpolation='none')
    plt.savefig(os.path.join(auxiliarPath, "check_SAX.png"))
    plt.close()

    samplePath = os.path.join(forNetPath, 'pred')
    if not os.path.exists(samplePath): pathlib.Path(samplePath).mkdir(parents=True, exist_ok=True)
    subject.img.save(os.path.join(samplePath, 'img.nii'), squeeze=True)
    
    print("New shape: {}".format(subject.img.shape))
    print("New spacing: {}".format(subject.img.spacing))


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',     type=str)
    parser.add_argument('--resPath',      type=str)
    parser.add_argument('--spacAfterCrop',type=str, required=True)
    parser.add_argument('--dataType',     type=str, required=True, help='Can be Exvivo, Cine, LGE')
    args = parser.parse_args()

    start = time.time()
    if args.dataType == "Cine":
        preproccessCineLGE(args.filePath, [160,160,20], args.spacAfterCrop, resPath=args.resPath)
    elif args.dataType == "LGE":
        preproccessCineLGE(args.filePath, [96,96,24], args.spacAfterCrop, resPath=args.resPath)
    elif args.dataType == "Exvivo":
        preproccessExvivo(args.filePath, resPath=args.resPath)
    else: 
        raise ValueError("Not implemented dataType")

    print("Total duration of processing: {} s ".format(time.time()-start))


if __name__ == '__main__':
    main()