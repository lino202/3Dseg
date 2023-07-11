'''This code get the topology for several datasets
In this case for MnMs, LGE and Exvivo

The conditions to work are that:
The dataset should be a unique max class presented in all samples
colors and labels list should be given for represent the priors

This code was studied in the preprocessed samples as the net will
be topologically retrained and it works with the data in this format.
This is clarify cause it might be discrepancies between the original 
topology and the one output when the data has been preprocessing
remember there is a resampling'''


import sys
import os
sys.path.append(os.path.join('/'.join(sys.path[0].split("/")[:-2])))
import utils.topo as topo
import argparse
import matplotlib.pyplot as plt
import time
import torchio as tio
import numpy as np
import pathlib
import pickle
import torch
import torch.nn.functional as F

priorMnMs = {
    (1,):   (1, 0, 0),
    (2,):   (1, 1, 0),
    (3,):   (1, 0, 0),
    (1, 2): (1, 0, 0),
    (1, 3): (1, 0, 0),
    (2, 3): (1, 1, 0)
}

priorExvivo = {
    (1,):   (1, 0, 0),
}


priorLGE = {
    (1,):   (1, 0, 0),
    (2,):   (1, 1, 0),
    (3,):   (1, 0, 0),
    (1, 2): (1, 0, 0),
    (1, 3): (1, 0, 0),
    (2, 3): (1, 1, 0)
}


priors = {'priorMnMs': priorMnMs, 
          'priorExvivo': priorExvivo,
          'priorLGE': priorLGE}


def main(priors):
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--dataPath',   type=str, required=True)
    parser.add_argument('--resPath',    type=str)
    parser.add_argument('--resName',    type=str)
    parser.add_argument('--phType',     type=str, required=True)
    parser.add_argument('--labels',     type=str, required=True, nargs='+')
    args = parser.parse_args()

    COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
              'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] #There is no more than ten priors never
    samples = sorted([x for x in os.listdir(os.path.join(args.dataPath))])
    print('There are {} samples in total'.format(len(samples)))

    if not os.path.exists(args.resPath): pathlib.Path(args.resPath).mkdir(parents=True, exist_ok=True)


    if "MnM" in args.dataPath:
        prior = priors['priorMnMs']
    elif "LGE" in args.dataPath:
        prior = priors['priorLGE']
    elif "Exvivo" in args.dataPath:
        prior = priors['priorExvivo']
    else:
        raise ValueError("Dataset not implemented")
    
    max_dims = [len(b) for b in prior.values()]
    prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    ph = {'0': topo.crip_wrapper, 'N': topo.trip_wrapper}
    results = np.ones((len(prior), len(samples), 3)) * np.nan
    
    for i, sample in enumerate(samples):

        print('Processing sample = {}'.format(sample))
        imgPath = os.path.join(args.dataPath, sample, "img.nii")
        mskPath = os.path.join(args.dataPath, sample, "msk.nii")

        # Get image and mask arrays
        subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        
        #we plot some info that can be used as double check (as we used the preprocessed samples here)
        print("Img min: {}, max: {}, shape: {}, spacing: {}".format(subject.img.data.min(), subject.img.data.max(), subject.img.shape, subject.img.spacing))
        print("Msk min: {}, max: {}, shape: {}, spacing: {}".format(subject.msk.data.min(), subject.msk.data.max(), subject.msk.shape, subject.msk.spacing))
            
        #Topo
        msk     = subject.msk.data[0]
        one_hot = F.one_hot(msk.long(), num_classes=int(msk.max().numpy()) + 1)
        msk     = one_hot.permute(3, 0, 1, 2).type(msk.type())
        
        barcodes = topo.getBarcodes(msk, prior, max_dims, ph, args.phType, False)
        for j in range(len(prior)):
            a , c = np.unique(barcodes[j][:,0], return_counts=True)
            results[j,i,tuple(a.astype(int))] = c

    with open(os.path.join(args.resPath, "{}.pickle".format(args.resName)), 'wb') as f:
        pickle.dump(results, f)
        
    plt.rcParams['figure.figsize'] = [15, 12]
    labels = args.labels
    x = np.arange(results.shape[1])
    for n,label in enumerate(labels):
        f, ax = plt.subplots(3,1)
        ax[0].scatter(x,results[n,:,0].T, c=COLORS[n], label=label)
        ax[1].scatter(x,results[n,:,1].T, c=COLORS[n], label=label)
        ax[2].scatter(x,results[n,:,2].T, c=COLORS[n], label=label)
        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        ax[2].legend(loc="upper right")
        ax[0].set_title("B0")
        ax[1].set_title("B1")
        ax[2].set_title("B2")
        ax[0].set_xlim([x.min(), x.max()+1])
        ax[1].set_xlim([x.min(), x.max()+1])
        ax[2].set_xlim([x.min(), x.max()+1])
        fig = plt.gcf()
        # plt.show()
        # plt.draw()
        fig.savefig(os.path.join(args.resPath, "{}_{}.png".format(args.resName, label)), dpi=100)
        plt.close()

if __name__ == '__main__':
    start = time.time()
    main(priors)
    print("Total duration processing: {} s ".format(time.time()-start))