"""General-purpose test script.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--results_dir' and save the results there.
"""
import os
from utils.options import TestOptions
from utils.util import mkdirs, mkdir, getStatistics
from data import create_dataloader
from models.model_unet import ModelUnet3D
import monai
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import utils.topo as topo
import pandas as pd
import pickle
import time
import pathlib


# Set prior_CINE_MnMs: class 1 is LV; 2 is MY; 3 is RV
# labelmap_CINE_MnMs = ['bg', 'lv', 'myo', 'rv']
prior_CINE_MnMs = {
    (1,):   (1, 0, 0),
    (2,):   (1, 1, 0), #Here maybe let open? (1,1,0) or close (1,0,0)
    (3,):   (1, 0, 0),
    (1, 2): (1, 0, 0),
    (1, 3): (2, 0, 0),
    (2, 3): (1, 1, 0)  #Here maybe let open? (1,1,0) or close (1,0,0)
}

# Set prior_LGE: class 1 is entire LV as it is binary
prior_roi = {(1,):   (1, 0, 0)}

# Set prior_LGE: N normal, P patological
prior_LGE_emidec = {
    "priorN" : {
        (1,):   (1, 0, 0),
        (2,):   (1, 1, 0),
        (1, 2): (1, 0, 0),
    },

    "priorP" : {
        (1,):   (1, 0, 0),
        (2,):   (1, 1, 0),
        (3,):   (1, 0, 0),
        (1, 2): (1, 0, 0),
        (1, 3): (1, 0, 0),
        (2, 3): (1, 1, 0)
    }
}

def main():
    
    # Get test options
    opt = TestOptions().parser.parse_args()
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    #Add results folders for plots and volumes
    plots_path    = os.path.join(opt.results_dir, opt.name, "plots")
    vols_path     = os.path.join(opt.results_dir, opt.name, "volumes")
    mkdirs([plots_path, vols_path])
    
    #Get dataloader
    test_dataloader = create_dataloader.create(opt, opt.phase)  # create a dataloader with given options
    nSamples        = len(test_dataloader.dataset)
    print('Testing with {} samples grouped in {} batches'.format(nSamples, len(test_dataloader)))
    
    #Get trained model
    model = ModelUnet3D(opt)       # create a Model
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.net.eval()               # affects layers like batchnorm and dropout.
    
    # Get prior
    generalPrior = globals()[opt.priorName]
    if opt.phThres < 0:
        phThres = None
    else: 
        phThres = opt.phThres
    
    gdsc = np.zeros((nSamples, opt.output_nc))
    hd   = np.zeros((nSamples, opt.output_nc))
    ts   = np.zeros(nSamples)
    be   = np.zeros(nSamples)
    print("Start testing ---------------------------------")
    for i, data in enumerate(test_dataloader):
        start_iter = time.time()
        model.set_input(data)  # unpack data from data loader
        sample  = pathlib.PureWindowsPath(model.path[0]).as_posix().split('/')[-1]
        
        if 'emidec' in opt.priorName:
            if 'P' in sample: 
                prior = generalPrior["priorP"]
            elif 'N' in sample:
                prior = generalPrior["priorN"]
            else: raise ValueError("Wrong prior name, should be normal or pato")
        else:
            prior = generalPrior
        
        #Determine result pred and binarize (one-hot pred)
        if opt.ph: 
            # Run topological post-processing
            model_TP = topo.multi_class_topological_post_processing(
                inputs=model.img, model=model.net, prior=prior,
                lr=1e-5, mse_lambda=1000,
                opt=torch.optim.Adam, num_its=100, construction='0', thresh=phThres, parallel=opt.phParallel
            )
            pred = model_TP(model.img)
        else:
            model.test()           # run inference
            pred = model.pred
        
        pred    = torch.softmax(pred, dim=1)
        pred    = pred.argmax(dim=1)
        one_hot = F.one_hot(pred.long(), num_classes=opt.output_nc)
        pred    = one_hot.permute(0, 4, 1, 2, 3).type(pred.type())
        pred = pred.to('cpu')
        
        one_hot = F.one_hot(model.msk.long(), num_classes=opt.output_nc)
        msk     = one_hot.permute(0, 4, 1, 2, 3).type(model.msk.type())
        msk  = msk.to('cpu')
        
        #Get img, name and affine. This serves to save plots and .nii
        affine  = model.affine.numpy()[0,:,:]
        img     = (model.img.to('cpu') + 1) / 2
        
        #Get gDSC, HD, BE and TS
        #there is an error on gDSC implementation as results has not shape [BxC]
        #For this reason we permute CxB and get background value in order to have the right values
        #Also it does not work when tensor are on cuda, they already submitted a PR.
        #TODO This should be checked or an issue should be raisen in https://github.com/Project-MONAI/MONAI
        gdsc[i,:] = monai.metrics.compute_generalized_dice(torch.permute(pred, (1,0,2,3,4)), torch.permute(msk, (1,0,2,3,4)), include_background=True)
        hd[i,:]   = monai.metrics.compute_hausdorff_distance(pred, msk, include_background=True)
        be[i]     = topo.BEmetric(pred[0,:,:,:], msk[0,:,:,:], prior, opt.phParallel)
        if be[i] == 0.: ts[i] = 1
        
        #Reverse one-hot encoded in mask and pred and get numpy arrays and get rid of the batch dim
        msk  = msk[0,:,:,:,:].argmax(dim=0).numpy().astype(float)
        pred = pred[0,:,:,:,:].argmax(dim=0).numpy().astype(float)   
        img  = img[0,0,:,:,:].numpy().astype(float) 
        
        #Save per volume results
        #Save images, we disregard the first and last image in the stack as usually are completly background
        nImgs = 5
        f, ax = plt.subplots(3,nImgs)
        sliceIdxs = np.linspace(0,img.shape[-1]-1,nImgs+2)
        sliceIdxs = np.round(sliceIdxs[1:-1]).astype(int)
        for j, s in enumerate(sliceIdxs):       
            ax[0,j].imshow(img[:,:,s], vmin=0, vmax=img.max())
            ax[1,j].imshow(msk[:,:,s], vmin=0, vmax=opt.output_nc-1)
            ax[2,j].imshow(pred[:,:,s], vmin=0, vmax=opt.output_nc-1)
        plt.savefig(os.path.join(plots_path, "{}.png".format(sample)))
        plt.close()
        
        #Save .nii 
        sample_vol_path = os.path.join(vols_path, sample)
        mkdir(sample_vol_path)
        predNifti = nib.Nifti1Image(pred, affine)
        nib.save(predNifti, os.path.join(sample_vol_path, "pred.nii"))
        
        #Print info
        print("Processed sample {}/{} took {} s".format(i+1, nSamples, time.time() - start_iter))
    
    #Save general results
    #Save per volumes parameters results  
    print("Saving results -------------------------------")
    resPath = os.path.join(opt.results_dir, opt.name)
    res_params = np.vstack((gdsc.T, hd.T, be, ts))
    res_dict = {}
    for i, index_name in enumerate(opt.res_excel_indexs):
        res_dict[index_name] = res_params[i,:]
    with open(os.path.join(resPath, "{}.pickle".format(opt.res_params_name)), 'wb') as f:
        pickle.dump(res_dict, f)
    
    #Save statiscal summary on excel with exp name
    exp_name = os.path.join(resPath, 'baseline' if not opt.ph else 'ph')
    
    res_dataframe = []
    for i, index_name in enumerate(opt.res_excel_indexs):
        if 'ts' != index_name:
            res_dataframe.append(getStatistics(res_params[i,:]))
        else:
            tmp = np.ones(9+1) * np.nan
            tmp[-1] = np.sum(ts) / nSamples
            res_dataframe.append(tmp)
    
    indexs  = ["{}_{}".format(exp_name, index) for index in opt.res_excel_indexs] 
    columns = ['mean', 'std', 'min', 'max', 'median', 'lowQuart', 'upQuart', 'lowWhisker', 'upWhisker', 'perc']
    df = pd.DataFrame(res_dataframe, index=indexs, columns=columns)
    
    if not os.path.exists(opt.res_excel):
        df.to_excel(opt.res_excel, sheet_name='sheet1')
    else:
        with pd.ExcelWriter(opt.res_excel, engine="openpyxl", mode='a',if_sheet_exists="overlay") as writer:
            startrow = writer.sheets['sheet1'].max_row
            df.to_excel(writer, sheet_name='sheet1', startrow=startrow, header=False)
    
 
if __name__ == '__main__':
    start = time.time()
    main()
    print("Total amount of time for processing {}".format(time.time() - start ))
