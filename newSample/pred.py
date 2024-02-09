"""General-purpose test script.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--results_dir' and save the results there.

This is different from the main pred.py as this combines the different 
nets obtained from the CV and average the results. 

Here, we suppose to have the preprocessed volumes and then
with another script we postprocess it, in order to have modularity
"""
import os
import torch
import torchio as tio
import time
import pathlib
import sys 

#TODO Not the most fancy thing to do I know
sys.path.append(os.path.join('/'.join(sys.path[0].split("/")[:-1])))
from utils.options import TestOptions
from utils.util import mkdir
from utils.priors import PRIOR_CINE, PRIOR_EXVIVO, PRIOR_LGE
from data import create_dataloader
from models.model_pred import ModelPred
import utils.topo as topo

def main():
    start = time.time()

    # Get test options
    opt = TestOptions().parser.parse_args()
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.gan        = False # 3DGAN cannot be used without a ground truth.

    #Get dataloader
    test_dataloader = create_dataloader.create(opt, opt.phase)  # create a dataloader with given options
    
    #Get trained nets folders
    net_folders = os.listdir(opt.results_dir)
    net_folders = [x for x in net_folders if opt.name in x]

    # Get prior
    if "CINE" in opt.priorName:     prior = PRIOR_CINE
    elif "EXVIVO" in opt.priorName: prior = PRIOR_EXVIVO
    elif "LGE" in opt.priorName:    prior = PRIOR_LGE
    else: raise ValueError("Wrong priorName")
    
    if opt.phThres < 0: phThres = None
    else: phThres = opt.phThres

    print("Start predicting ---------------------------------")

    input = next(iter(test_dataloader))
    pred_t = torch.zeros((1, opt.output_nc, *opt.patch_size))

    for n, net_folder in enumerate(net_folders):
        start_iter = time.time()
        opt.name = net_folder
        vols_path     = os.path.join('/'.join(opt.root_path.split('/')[:-1]), opt.name, "volumes_{}_phconst{}".format('baseline' if not opt.ph else 'ph',  opt.phConstruction))
        mkdir(vols_path)

        #Get trained model
        model = ModelPred(opt)       # create a Model
        model.setup(opt)             # regular setup: load and print networks; create schedulers
        model.net.eval()             # affects layers like batchnorm and dropout.

        model.set_input(input)  # unpack data from data loader
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
        pred = pred.detach().to('cpu')
        pred_t = pred_t + pred

        #Print info
        print("Processed sample with net {}/{} took {} s".format(n+1, len(net_folders), time.time() - start_iter))

    pred_t = pred_t / len(net_folders)
    pred_t = pred_t.argmax(dim=1)
    affine  = model.affine.numpy()[0,:,:]

    #Save per volume results
    #Save .nii 
    pred_t = tio.LabelMap(tensor=pred_t, affine=affine)
    pred_t.save(os.path.join('/'.join(opt.root_path.split('/')[:-1]), 'pred_total.nii'), squeeze=True)

    print("Total predicting time was {} s".format(time.time() - start))

 
if __name__ == '__main__':
    main()
