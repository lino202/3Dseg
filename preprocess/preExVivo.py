'''This preprocess the Ex vivo dataset:
Create with B0 images from the DWI data and T1, T2 sequences when available. The data come from
an Standfor publicly available dataset and from projects Brav3 and Cardioprint. 
Mi is present in some of the samples labelled as P for patological and N for normal'''


import os
import argparse
import matplotlib.pyplot as plt
import time
import torchio as tio
import numpy as np
import pathlib
import traceback

def process(dataPath, size, plotPath, resPath=None, save=False):

    samples = sorted([x for x in os.listdir(dataPath)])
    print('There are {} samples in total'.format(len(samples)))

    for sample in samples:

        print('Processing sample = {}'.format(sample))
        imgPath = os.path.join(dataPath, sample, "img.nii")
        mskPath = os.path.join(dataPath, sample, "msk.nii")

        # If the short-axis image file exists, read the data and perform processing
        if (os.path.isfile(imgPath) and os.path.isfile(mskPath)):

            try:
                # Get image and mask arrays
                subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
                
                #Reshape for having isotropic voxels
                #Compliant with real world xyz coordinates (modifies affine matrix). see Torchio/nibabel docs
                sw = subject.img.shape[1] * subject.img.spacing[0] / size[0]   # there's an extra dim in the shape
                sh = subject.img.shape[2] * subject.img.spacing[1] / size[1]
                sd = subject.img.shape[3] * subject.img.spacing[2] / size[2]
                trans = tio.Resample((sw, sh, sd), image_interpolation='linear', label_interpolation='nearest') #Try lanczos?
                subject.img = trans(subject.img)
                subject.msk = trans(subject.msk)
                
                transNorm = tio.ZNormalization()
                subject.img = transNorm(subject.img)
                
                #Separate ED and ES 3D arrays
                results = {}
                results["img"] = subject.img.data.numpy()[0,:,:,:]
                results["msk"] = subject.msk.data.numpy()[0,:,:,:]
                
                # Save and/or plot 3D ED and ES arrays
                if plotPath != None:
                    tmp = results["msk"].nonzero()[2]
                    s = ((np.max(tmp) - np.min(tmp)) / 2) + np.min(tmp)
                    s = int(np.round(s))
                    f, ax = plt.subplots(1,2)
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
    parser.add_argument('--trainPath',    type=str)
    parser.add_argument('--testPath',     type=str)
    parser.add_argument('--valPath',      type=str)
    parser.add_argument('--trainResPath', type=str)
    parser.add_argument('--testResPath',  type=str)
    parser.add_argument('--valResPath',   type=str)
    parser.add_argument('--plotPath',     type=str)
    parser.add_argument('--size',         type=int, required=True, nargs=3)
    args = parser.parse_args()
    
    if hasattr(args, 'plotPath'):
        plotPath = args.plotPath
    else:
        plotPath = None
    
    startTot = time.time()
    
    #Training samples
    if args.trainPath != None:
        start = time.time()
        print("Processing Training samples at {}".format(args.trainPath))
        process(args.trainPath, args.size, plotPath, resPath=args.trainResPath, save=True)
        print("Total duration of Training processing: {} s ".format(time.time()-start))
    
    #Validation samples
    if args.valPath != None:
        start = time.time()
        print("Processing Validation samples at {}".format(args.valPath))
        process(args.valPath, args.size, plotPath, resPath=args.valResPath, save=True)
        print("Total duration of Validation processing: {} s ".format(time.time()-start))
    
    #Testing samples
    if args.testPath != None:
        start = time.time()
        print("Processing Testing samples at {}".format(args.testPath))
        process(args.testPath, args.size, plotPath, resPath=args.testResPath, save=True)
        print("Total duration of Testing processing: {} s ".format(time.time()-start))
    
    print("Total duration processing: {} s ".format(time.time()-startTot))
    

if __name__ == '__main__':
    main()