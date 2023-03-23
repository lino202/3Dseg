'''This preprocess the Emidec dataset:
Initially neither slicer neither itk-snap read the files correctly
Also only the training data gt is available so we randomly separate all the files

Also we cannot crop Roi in here with cine so we gen LV mask sets in order to train
a Roi detection network. So the code work in two stages the first is to generate the sets
for train, test and val for a Roi segmentation network. Then as the samples distribution was save,
all the predictions computed by the Roi segmentation network are use for cropping and finally getting
the train, test and val for training. The last label of no-reflow is together with the myocardial
infarction one for simplicity in the topology prior determination.
'''

import os
import argparse
import matplotlib.pyplot as plt
import time
import torchio as tio
import numpy as np
import pathlib
import traceback
import random
import pickle
from utilsPre import cropSubjectPred


def getIdxs4Datasets(nPato, nHe):
    # Get 65, 20 and 15 % for train test and val
    idxs = np.arange(nPato)
    trainPatoIdxs = random.sample(list(idxs), int(np.round(nPato*0.65)))
    tmp = np.logical_not(np.isin(idxs, trainPatoIdxs)).nonzero()[0]
    testPatoIdxs = random.sample(list(tmp), int(np.round(nPato*0.2)))
    valPatoIdxs = list(np.logical_not(np.isin(idxs, trainPatoIdxs + testPatoIdxs)).nonzero()[0])
    
    idxs = np.arange(nHe)
    trainHeIdxs = random.sample(list(idxs), int(np.round(nHe*0.65)))
    tmp = np.logical_not(np.isin(idxs, trainHeIdxs)).nonzero()[0]
    testHeIdxs = random.sample(list(tmp), int(np.round(nHe*0.2)))
    valHeIdxs = list(np.logical_not(np.isin(idxs, trainHeIdxs + testHeIdxs)).nonzero()[0])
    
    return trainPatoIdxs, testPatoIdxs, valPatoIdxs, trainHeIdxs, testHeIdxs, valHeIdxs

def process(dataPath, size, samples, roi=False, plotPath=None, resPath=None, roiPredPath=None, save=False):

    print('There are {} samples in total'.format(len(samples)))
    
    for sample in samples:

        print('Processing sample = {}'.format(sample))
        imgPath = os.path.join(dataPath, sample, "Images", "{}.nii.gz".format(sample))
        mskPath = os.path.join(dataPath, sample, "Contours",   "{}.nii.gz".format(sample))

        # If the short-axis image file exists, read the data and perform processing
        if (os.path.isfile(imgPath) and os.path.isfile(mskPath)):

            try:
                # Get image and mask arrays
                subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
                
                # Crop if rois are available
                if not roi and roiPredPath != None:
                    #The prediction need to be in the same space than img and msk
                    pred    = tio.ScalarImage(os.path.join(roiPredPath, sample, "pred.nii"))
                    sw = pred.shape[1] * pred.spacing[0] / subject.img.shape[1]
                    sh = pred.shape[2] * pred.spacing[1] / subject.img.shape[2]
                    sd = pred.shape[3] * pred.spacing[2] / subject.img.shape[3]
                    trans = tio.Resample((sw, sh, sd))
                    pred = trans(pred)
                    subject = cropSubjectPred(subject, pred)
                
                #Reshape for having isotropic number of voxels
                #Compliant with real world xyz coordinates (modifies affine matrix). see Torchio/nibabel docs
                sw = subject.img.shape[1] * subject.img.spacing[0] / size[0]   # there's an extra dim in the shape
                sh = subject.img.shape[2] * subject.img.spacing[1] / size[1]
                sd = subject.img.shape[3] * subject.img.spacing[2] / size[2]
                trans = tio.Resample((sw, sh, sd))
                subject.img = trans(subject.img)
                subject.msk = trans(subject.msk)
                
                transNorm = tio.ZNormalization()
                subject.img = transNorm(subject.img)
                
                results = {}
                results["img"] = subject.img.data.numpy()[0,:,:,:]
                results["msk"] = subject.msk.data.numpy()[0,:,:,:]
                
                if roi:
                    results["msk"][results["msk"]>0] = 1 
                else:
                    results["msk"][results["msk"]>3] = 3 
                
                # Save and/or plot 3D ED and ES arrays
                if plotPath != None:
                    f, ax = plt.subplots(1,2)
                    s = int(np.round(results["img"].shape[-1]/2))
                    ax[0].imshow(results["img"][:,:,s])
                    ax[1].imshow(results["msk"][:,:,s])
                    plt.savefig(os.path.join(plotPath, "{}.png".format(sample)))
                    plt.close()
                
                if save:
                    samplePath = os.path.join(resPath, sample)
                    if not os.path.exists(samplePath): pathlib.Path(samplePath).mkdir(parents=True, exist_ok=True)
                    for key in results:
                        tioImg   = tio.ScalarImage(tensor=results[key][np.newaxis,:], affine=subject.img.affine)
                        tioImg.save(os.path.join(samplePath, '{}.nii'.format(key)), squeeze=True)
            except Exception as e:
                print("SAMPLE {} WAS NOT PROCESSED AS IT FINISHED WITH EXCEPTION {}".format(sample, traceback.format_exc()))
                
        else:
            print('There is no image or mask volume file for sample {}'.format(sample))


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--rootPath',    type=str)
    parser.add_argument('--resPath',     type=str)
    parser.add_argument('--plotPath',     type=str)
    parser.add_argument('--roiPredPath',type=str)
    parser.add_argument('--samplesDistPath',type=str)
    parser.add_argument('--roi',         action='store_true')
    parser.add_argument('--size',        type=int, required=True, nargs=3)
    args = parser.parse_args()
    
    if hasattr(args, 'plotPath'):
        plotPath = args.plotPath
    else:
        plotPath = None
    
    #Get samples distribution
    if not hasattr(args, 'samplesDistPath'):
        samplesPato = sorted([x for x in os.listdir(args.rootPath) if not '.txt' in x and 'P' in x])
        samplesHe = sorted([x for x in os.listdir(args.rootPath) if not '.txt' in x and 'N' in x])
        print('There are {} healthy samples in total'.format(len(samplesHe)))
        print('There are {} pato samples in total'.format(len(samplesPato)))
        
        #separate healthy and pato samples equivalently
        nPato = len(samplesPato)
        nHe = len(samplesHe)
        
        trainPatoIdxs, testPatoIdxs, valPatoIdxs, trainHeIdxs, testHeIdxs, valHeIdxs = getIdxs4Datasets(nPato, nHe)
        trainPatoSamples = [samplesPato[idx] for idx in trainPatoIdxs]
        trainHeSamples   = [samplesHe[idx] for idx in trainHeIdxs]
        trainSamples     = trainPatoSamples + trainHeSamples
        
        testPatoSamples = [samplesPato[idx] for idx in testPatoIdxs]
        testHeSamples   = [samplesHe[idx] for idx in testHeIdxs]
        testSamples     = testPatoSamples + testHeSamples
        
        valPatoSamples = [samplesPato[idx] for idx in valPatoIdxs]
        valHeSamples   = [samplesHe[idx] for idx in valHeIdxs]
        valSamples     = valPatoSamples + valHeSamples
    
        #Save distribution just in case
        samplesdict = {"trainSamples" : trainSamples, "testSamples" : testSamples, "valSamples" : valSamples}
        with open(os.path.join(args.resPath, "samples_distribution.pickle"), 'wb') as f:
            pickle.dump(samplesdict, f)
            
    else:
        with open(args.samplesDistPath, 'rb') as handle:
            samplesdict = pickle.load(handle)
            
        trainSamples = samplesdict ["trainSamples"]
        testSamples = samplesdict ["testSamples"]
        valSamples = samplesdict ["valSamples"]
           
    
    print("Processing Training samples")
    resPathTrain = os.path.join(args.resPath , "train")
    if not os.path.exists(resPathTrain): pathlib.Path(resPathTrain).mkdir(parents=True, exist_ok=True)
    process(args.rootPath, args.size, trainSamples, args.roi, plotPath=plotPath,  resPath=resPathTrain, roiPredPath=args.roiPredPath, save=True)
    
    print("Processing Testing samples")
    resPathTest = os.path.join(args.resPath , "test")
    if not os.path.exists(resPathTest): pathlib.Path(resPathTest).mkdir(parents=True, exist_ok=True)
    process(args.rootPath, args.size, testSamples, args.roi, plotPath=plotPath, resPath=resPathTest, roiPredPath=args.roiPredPath, save=True)
    
    print("Processing Validation samples")
    resPathVal = os.path.join(args.resPath , "val")
    if not os.path.exists(resPathVal): pathlib.Path(resPathVal).mkdir(parents=True, exist_ok=True)
    process(args.rootPath, args.size, valSamples, args.roi, plotPath=plotPath, resPath=resPathVal, roiPredPath=args.roiPredPath, save=True)

    
if __name__ == '__main__':
    start = time.time()
    main()
    print("Total duration processing: {} s ".format(time.time()-start))