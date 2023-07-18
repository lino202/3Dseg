import os
from utils.options import TestOptions
from utils.util import mkdir
from data import create_dataloader
from models.model_pred import ModelPred
import matplotlib.pyplot as plt
import nibabel as nib
import time
import numpy as np
import pathlib


def main():
    # Get test options
    opt            = TestOptions().parser.parse_args()
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.gan        = True

    plots_path = os.path.join(opt.results_dir, opt.name, "plots")
    mkdir(plots_path)
    
    #Get dataloader
    pred_dataloader = create_dataloader.create(opt, opt.phase)  # create a dataloader with given options
    nSamples        = len(pred_dataloader.dataset)
    print('Testing with {} samples grouped in {} batches'.format(nSamples, len(pred_dataloader)))
    
    #Get trained model
    model = ModelPred(opt)       # create a Model
    model.setup(opt)             # regular setup: load and print networks; create schedulers
    model.net.eval()             # affects layers like batchnorm and dropout.
    
    print("Start predicting ---------------------------------")
    for i, data in enumerate(pred_dataloader):
        start_iter = time.time()
        model.set_input(data)  # unpack data from data loader
        sample  = pathlib.PureWindowsPath(model.path[0]).as_posix().split('/')[-1]
        
        model.test()           # run inference
        pred = model.pred.to('cpu').numpy()
        img  = model.img.to('cpu').numpy()    # this is the msk
        pred = create_dataloader.unitNorm(np.squeeze(pred))
        img  = create_dataloader.unitNorm(np.squeeze(img))
        affine  = model.affine.numpy()[0,:,:]
        
        #We have 3D arrays [0,1] we save the img.nii and save the plots (msk.nii is not resaved)
        #Save plots
        nImgs = 5
        f, ax = plt.subplots(2,nImgs)
        sliceIdxs = np.linspace(0,img.shape[-1]-1,nImgs+2)
        sliceIdxs = np.round(sliceIdxs[1:-1]).astype(int)
        for j, s in enumerate(sliceIdxs):       
            ax[0,j].imshow(img[:,:,s], vmin=0, vmax=1)
            ax[1,j].imshow(pred[:,:,s], vmin=0, vmax=1)
        plt.savefig(os.path.join(plots_path, "{}.png".format(sample)))
        plt.close()
        
        #Save .nii 
        sample_vol_path = os.path.join(opt.root_path, sample)
        predNifti = nib.Nifti1Image(pred, affine)
        nib.save(predNifti, os.path.join(sample_vol_path, "img.nii"))
        
        #Print info
        print("Processed sample {}/{} took {} s".format(i+1, nSamples, time.time() - start_iter))
    
    
if __name__ == '__main__':
    start = time.time()
    main()
    print("Total duration processing: {} s ".format(time.time()-start))