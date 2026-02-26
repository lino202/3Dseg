from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.persistent_homology import nnUNetPredictorPH
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.priors import PRIOR_LGE



def predict():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help="Dataset name")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint name")
    parser.add_argument('--resFolderName', type=str, help="Checkpoint name")
    parser.add_argument('--ph', default=False, action='store_true')
    parser.add_argument('--folds', type=int, default=5, help="Number of folds to predict on. Default is 5 for 5-fold cross validation.")
    args = parser.parse_args()

    # instantiate the nnUNetPredictor
    if not args.ph:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
    else:
        predictor = nnUNetPredictorPH(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
            priors = {'D8': PRIOR_LGE, 'MX': PRIOR_LGE, 'default': PRIOR_LGE} # This is like this due to how we implemented the priors selection inside the predictor only MX is used for the LGE dataset
        )

    # In this case we want to predict the data in or validation set here, WITH and WITHOUT persistent homology PH. 
    # So we use every fold for prediction of its corresponding validation set. 
    # This is not the typical use case for nnUNetPredictor but it is a good way to compare the results with and without PH. 
    samples = sorted([x for x in os.listdir(join(nnUNet_raw, '{}/imagesTr'.format(args.dataset_name)))])
    splits = load_json(join(nnUNet_preprocessed, '{}/splits_final.json'.format(args.dataset_name)))
    resultsFolder = join(nnUNet_results, args.resFolderName)
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    for fold in range(args.folds):

        print(f"Predicting fold {fold}...")
        samplesInFold = sorted([x for x in samples if x.split('_0000')[0] in splits[fold]['val']])
    
        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, '{}/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres'.format(args.dataset_name)),
            use_folds=(fold,),
            checkpoint_name='{}.pth'.format(args.checkpoint),
        )

        # Here we segment one by one with PH to avoid getting in the confusion of multiprocessing, might be slow but it is less error prone
        #  if it is unbearable the slowness, calling predict_from_files might  be an option
        for sample in samplesInFold:

            start = time.time()
            img, props = SimpleITKIO().read_images([join(nnUNet_raw, '{}/imagesTr'.format(args.dataset_name), sample)]) # this is I/O used
            ret = predictor.predict_single_npy_array(img, 
                                                    props, 
                                                    None, 
                                                    join(resultsFolder, sample.split('_0000')[0]), 
                                                    True)
            print("The time for sample {} was {} s".format(sample, time.time()-start))

if __name__ == '__main__': 
    start = time.time()
    predict()
    print("Total time was {} s".format(time.time()-start))