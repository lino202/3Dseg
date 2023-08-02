'''Here we transform masks with torchio combining elastic and affine transforms'''

# from utilsPre import myRandomElasticDeformation
# transform = myRandomElasticDeformation(num_control_points=(7,7,7), max_displacement=(12.,12.,0.), locked_borders=2)
# newArr = np.zeros(msk.shape)
# for s in range(msk.shape[2]):
#     imgMsk = tio.LabelMap(tensor=msk.data.numpy()[...,s][...,np.newaxis])
#     imgMskTrans = transform(imgMsk)
#     newArr[...,s] = imgMskTrans.data.numpy()[...,0]
# mskTrans = tio.LabelMap(tensor=newArr, affine=msk.affine)

import os
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import pathlib
import torchio as tio


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--rootPath',    type=str)
    parser.add_argument('--resPath',     type=str)
    parser.add_argument('--plotPath',     type=str)
    parser.add_argument('--maxDisp',     type=float)
    args = parser.parse_args()
    
    if not os.path.exists(args.resPath): pathlib.Path(args.resPath).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(args.plotPath): pathlib.Path(args.plotPath).mkdir(parents=True, exist_ok=True)
    
    samples = sorted([x for x in os.listdir(args.rootPath) if not '.txt' in x])
    print('There are {} samples in total'.format(len(samples)))
    
    transAffine = tio.RandomAffine(
        scales=0.2,
        degrees=(0.,0.,180.),
        translation=(10.,10.,0.),
        image_interpolation='nearest'
    )
    
    transElastic = tio.RandomElasticDeformation(
        num_control_points=(5,5,5), 
        max_displacement=(args.maxDisp,args.maxDisp,0.), 
        locked_borders=2,
        image_interpolation='nearest'
    )
    
    for i, sample in enumerate(samples):
        print('Processing sample = {}'.format(sample))
        mskPath = os.path.join(args.rootPath, sample, "msk.nii")
        
        msk = tio.LabelMap(mskPath)

        #Passing the real affine results is weird results, then I transform and identity affine image
        #and posteriorly add the right affine
        mskTrans = transElastic(msk.data)
        mskTrans = transAffine(mskTrans)
        mskTrans = tio.LabelMap(tensor=mskTrans, affine=msk.affine)
      
        f, ax = plt.subplots(2,3)
        s2 = int(np.round(msk.shape[-1]/2))
        s1 = int(np.round(msk.shape[-2]/2))
        s0 = int(np.round(msk.shape[-3]/2))
        ax[0,0].imshow(msk.data.numpy()[0,:,:,s2], vmin=0, vmax=3, interpolation='none')
        ax[0,1].imshow(msk.data.numpy()[0,:,s1,:], vmin=0, vmax=3, interpolation='none')
        ax[0,2].imshow(msk.data.numpy()[0,s0,:,:], vmin=0, vmax=3, interpolation='none')
        ax[1,0].imshow(mskTrans.data.numpy()[0,:,:,s2], vmin=0, vmax=3, interpolation='none')
        ax[1,1].imshow(mskTrans.data.numpy()[0,:,s1,:], vmin=0, vmax=3, interpolation='none')
        ax[1,2].imshow(mskTrans.data.numpy()[0,s0,:,:], vmin=0, vmax=3, interpolation='none')
        plt.savefig(os.path.join(args.plotPath, "{}.png".format(sample))) if args.plotPath else plt.show()
        plt.close()
        
        resSamplePath = os.path.join(args.resPath, "{}_deform".format(sample))
        if not os.path.exists(resSamplePath): pathlib.Path(resSamplePath).mkdir(parents=True, exist_ok=True)
        mskTrans.save(os.path.join(resSamplePath, "msk.nii"), squeeze=True)

    
if __name__ == '__main__':
    start = time.time()
    main()
    print("Total duration processing: {} s ".format(time.time()-start))