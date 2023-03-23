'''This preprocess the MnMs dataset:
Slicer does not read well 4D nifti so if you loaded in python nibabel or torchio and save the 3D of ED or ES
you will see the array correctly. The location in space xyz of the volume changes. 
The affine matrix or spatial location are respected with this code as torchio modifies the affine in consequence
to the transform applied
SimpleItk must be 2.0.2 otherwise an error on some images happens ITK ERROR: ITK only supports orthonormal direction cosines. 
No orthonormal definition found!
    
Problems with 
testing A7E4J0 in roi cropping as movement artifacts in the corners (disregarded)
testing E3L8U8 weird original segmentation but it is use'''


import os
import argparse
import time
import torchio as tio
import numpy as np
from utilsPre import getEXFromMask, cropSubjectCINE
import pathlib
import traceback

def process(dataPath, size, plotPath, resPath=None, save=False):

    samples = sorted([x for x in os.listdir(dataPath)])
    print('There are {} samples in total'.format(len(samples)))

    for sample in samples:

        print('Processing sample = {}'.format(sample))
        imgPath = os.path.join(dataPath, sample, "{}_sa.nii.gz".format(sample))
        mskPath = os.path.join(dataPath, sample, "{}_sa_gt.nii.gz".format(sample))

        # If the short-axis image file exists, read the data and perform processing
        if (os.path.isfile(imgPath) and os.path.isfile(mskPath)):

            try:
                # Get image and mask arrays
                subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
                edes = getEXFromMask(subject.msk.data.numpy(), sample)

                # Crop both 4D arrays
                # This maintain real world coordinates xyz
                subject = cropSubjectCINE(subject, edes)
                
                #Reshape for having isotropic voxels
                #Compliant with real world xyz coordinates (modifies affine matrix). see Torchio/nibabel docs
                sw = subject.img.shape[1] * subject.img.spacing[0] / size[0]   # there's an extra dim in the shape
                sh = subject.img.shape[2] * subject.img.spacing[1] / size[1]
                sd = subject.img.shape[3] * subject.img.spacing[2] / size[2]
                trans = tio.Resample((sw, sh, sd), image_interpolation='linear', label_interpolation='nearest') #Try lanczos?
                subject.img = trans(subject.img)
                subject.msk = trans(subject.msk)
                
                #Separate ED and ES 3D arrays
                results = {}
                results["img_ED"] = subject.img.data.numpy()[edes[0],:,:,:]
                results["msk_ED"] = subject.msk.data.numpy()[edes[0],:,:,:]
                results["img_ES"] = subject.img.data.numpy()[edes[1],:,:,:]
                results["msk_ES"] = subject.msk.data.numpy()[edes[1],:,:,:]
                
                # Save and/or plot 3D ED and ES arrays
                if plotPath != None:
                    import matplotlib.pyplot as plt
                    f, ax = plt.subplots(2,2)
                    s = int(np.round(results["img_ED"].shape[-1]/2))
                    ax[0,0].imshow(results["img_ED"][:,:,s])
                    ax[0,1].imshow(results["msk_ED"][:,:,s])
                    ax[1,0].imshow(results["img_ES"][:,:,s])
                    ax[1,1].imshow(results["msk_ES"][:,:,s])
                    plt.savefig(os.path.join(plotPath, "{}_roi.png".format(sample)))
                    plt.close()
                
                if save:
                    samplePath = os.path.join(resPath, sample)
                    for key in results:
                        tioImg   = tio.ScalarImage(tensor=results[key][np.newaxis,:], affine=subject.img.affine)
                        moment   = key.split('_')[-1]
                        dataType = key.split('_')[0]
                        filePath = "{}_{}".format(samplePath, moment)
                        if not os.path.exists(filePath): pathlib.Path(filePath).mkdir(parents=True, exist_ok=True)
                        tioImg.save(os.path.join(filePath, '{}.nii'.format(dataType)), squeeze=True)
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