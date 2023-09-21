"""General-purpose test script.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--results_dir' and save the results there.
"""
import os
from utils.options import TestOptions
from utils.util import mkdir, Logger
from utils.priors import PRIOR_CINE, PRIOR_EXVIVO, PRIOR_LGE
from data import create_dataloader
from models.model_pred import ModelPred
import torch
import torchio as tio
import utils.topo as topo
import time
import pathlib
import sys

def main():
    start = time.time()

    # Get test options
    opt = TestOptions().parser.parse_args()
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.gan        = False # 3DGAN cannot be used without a ground truth.
    sys.stdout = Logger(os.path.join(opt.results_dir, opt.name, "pred_output_{}.out".format('baseline' if not opt.ph else 'ph')))

    #Add results folders for plots and volumes
    vols_path     = os.path.join(opt.results_dir, opt.name, "volumes_{}_phconst{}".format('baseline' if not opt.ph else 'ph',  opt.phConstruction))
    mkdir(vols_path)
    
    #Get dataloader
    test_dataloader = create_dataloader.create(opt, opt.phase)  # create a dataloader with given options
    nSamples        = len(test_dataloader.dataset)
    print('Testing with {} samples grouped in {} batches'.format(nSamples, len(test_dataloader)))
    
    #Get trained model
    model = ModelPred(opt)       # create a Model
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.net.eval()               # affects layers like batchnorm and dropout.
    
    # Get prior
    if "CINE" in opt.priorName:     prior = PRIOR_CINE
    elif "EXVIVO" in opt.priorName: prior = PRIOR_EXVIVO
    elif "LGE" in opt.priorName:    prior = PRIOR_LGE
    else: raise ValueError("Wrong priorName")
    
    if opt.phThres < 0: phThres = None
    else: phThres = opt.phThres

    print("Start predicting ---------------------------------")

    for i, data in enumerate(test_dataloader):
        start_iter = time.time()
        model.set_input(data)  # unpack data from data loader
        sample  = pathlib.PureWindowsPath(model.path[0]).as_posix().split('/')[-1]
        sample_vol_path = os.path.join(vols_path, sample)
        mkdir(sample_vol_path)
        print("Predicting sample {}".format(sample))

        #Determine result pred
        if opt.ph: 
            # Run topological post-processing
            model_TP = topo.multi_class_topological_post_processing(
                inputs=model.img, model=model.net, prior=prior,
                lr=1e-5, mse_lambda=1000,
                opt=torch.optim.Adam, num_its=100, construction=opt.phConstruction, 
                thresh=phThres, parallel=opt.phParallel, saveCombosPath=sample_vol_path,
                saveLogitsPath=sample_vol_path
            )
            pred = model_TP(model.img)
        else:
            model.test()           # run inference
            pred = model.pred

        pred = torch.softmax(pred, dim=1)
        pred = pred.argmax(dim=1)
        pred = pred.to('cpu')
        affine  = model.affine.numpy()[0,:,:]
  
        #Save per volume results
        #Save .nii 
        pred = tio.LabelMap(tensor=pred, affine=affine)
        pred.save(os.path.join(sample_vol_path, 'pred.nii'), squeeze=True)
        
        #Print info
        print("Processed sample {}/{} took {} s".format(i+1, nSamples, time.time() - start_iter))
    
    print("Total training time was {} s".format(time.time() - start))
 
if __name__ == '__main__':
    main()
