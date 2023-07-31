'''This preprocess the Exvivo dataset:
Here the shapes and spacings are ALMOST isotropic so it is easy than the others datasets
this is why we do not crop anything here and just we resample

Final shape is 128x128x128 and spacing 1x1x0.9375 mm3
Min [128. 128. 128.] Median [128. 128. 128.] Max [128. 128. 128.] shapes
Min [1.     1.     0.9375] Median [1.     1.     0.9375] Max [1.     1.     0.9375] spacings
'''

import os
import argparse
import time
import torchio as tio
import numpy as np
import pickle
import pathlib
import torch.nn.functional as F
import matplotlib.pyplot as plt


def proccess(dataPath, resPath=None, save=False):

    samples = sorted([x for x in os.listdir(dataPath)])
    print('There are {} samples in total'.format(len(samples)))
    
    newshapes   = np.zeros([len(samples),3])
    newspacings = np.zeros([len(samples),3])
    plotPath    = os.path.join(resPath, "plots_preprocessed")
    if not os.path.exists(plotPath): pathlib.Path(plotPath).mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):

        print('Processing sample = {}'.format(sample))
        imgPath = os.path.join(dataPath, sample, "img.nii")
        mskPath = os.path.join(dataPath, sample, "msk.nii")

        # Get image and mask arrays
        subjectOri = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        subjectOri.check_consistent_attribute('spacing') 
        subjectOri.check_consistent_attribute('affine')
        subjectOri.check_consistent_attribute('shape')

        #Keep biggest island
        trans = tio.KeepLargestComponent()
        tmp = subjectOri.msk.data[0]
        tmp = F.one_hot(tmp.long(), num_classes=int(subjectOri.msk.data.max())+1)
        tmp = tmp.permute(3, 0, 1, 2).type(tmp.type())
        for fg in range(1,tmp.shape[0]):
            subjectMsk = tio.Subject(msk=tio.LabelMap(tensor=tmp[fg][np.newaxis,:]))
            subjectMsk = trans(subjectMsk)
            tmp[fg] = subjectMsk.msk.data
        tmp = tmp.argmax(dim=0)
        subjectOri.msk.data[0] = tmp

        #Resample and crop/padding is not neccesary here as samples are 128x128x128 shape and 1x1x0.9375 spacing

        #Normalize - not neccesary to save for invert, this should be last
        trans = tio.ZNormalization()
        subject = trans(subjectOri)

        newshapes[i,:] = subject.img.shape[-3:]
        newspacings[i,:] = subject.img.spacing

        results = {}
        results["img_Ori"] = subjectOri.img.data.numpy()[0]
        results["msk_Ori"] = subjectOri.msk.data.numpy()[0]
        results["img"] = subject.img.data.numpy()[0]
        results["msk"] = subject.msk.data.numpy()[0]

        # Save
        if save:
            f, ax = plt.subplots(2,2)
            sOri = int(np.round(results["img_Ori"].shape[-1]/2))
            s = int(np.round(results["img"].shape[-1]/2))
            ax[0,0].imshow(results["img_Ori"][:,:,sOri], vmin=results["img_Ori"].min(), vmax = results["img_Ori"].max(), interpolation='none')
            ax[0,1].imshow(results["msk_Ori"][:,:,sOri], vmin=0, vmax=1, interpolation='none')
            ax[1,0].imshow(results["img"][:,:,s], vmin=results["img"].min(), vmax = results["img"].max(), interpolation='none')
            ax[1,1].imshow(results["msk"][:,:,s], vmin=0, vmax=1, interpolation='none')
            plt.savefig(os.path.join(plotPath, "{}.png".format(sample)))
            plt.close()
        
            samplePath = os.path.join(resPath, "vols_preprocessed", sample)
            if not os.path.exists(samplePath): pathlib.Path(samplePath).mkdir(parents=True, exist_ok=True)
            subject.img.save(os.path.join(samplePath, 'img.nii'), squeeze=True)
            subject.msk.save(os.path.join(samplePath, 'msk.nii'), squeeze=True)

            #Here we do not save any subject as preprocessing history
            #would not include any spatial trnasformation

    res_dict = {"voxels": newshapes, "spacings": newspacings}
    with open(os.path.join(resPath, "shapes_spacing_preprocessed.pickle"), 'wb') as f:
        pickle.dump(res_dict, f)
    
    print("Min {} Median {} Max {} shapes".format(np.min(newshapes,0), np.median(newshapes,0), np.max(newshapes,0)))
    print("Min {} Median {} Max {} spacings".format(np.min(newspacings,0), np.median(newspacings,0), np.max(newspacings,0)))

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',   type=str)
    parser.add_argument('--resPath',    type=str)
    args = parser.parse_args()

    start = time.time()

    print("Processing samples at {}".format(args.filePath))
    proccess(args.filePath, resPath=args.resPath, save=True)
    print("Total duration of processing: {} s ".format(time.time()-start))


if __name__ == '__main__':
    main()