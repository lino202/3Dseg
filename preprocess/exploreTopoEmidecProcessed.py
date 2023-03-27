
import sys
import os
sys.path.append(os.path.join('/'.join(sys.path[0].split("/")[:-1])))
import utils.topo as topo
import argparse
import matplotlib.pyplot as plt
import time
import nibabel as nib
import numpy as np
import pickle
import torch
import torch.nn.functional as F

priorN = {
    (1,):   (1, 0, 0),
    (2,):   (1, 1, 0),
    (1, 2): (1, 0, 0),
}

priorP = {
    (1,):   (1, 0, 0),
    (2,):   (1, 1, 0),
    (3,):   (1, 0, 0),
    (1, 2): (1, 0, 0),
    (1, 3): (1, 0, 0),
    (2, 3): (1, 1, 0)
}

def main(priorN, priorP):
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--rootPath',    type=str)
    parser.add_argument('--resPath',    type=str)
    parser.add_argument('--resName',    type=str)
    args = parser.parse_args()

    samples = sorted([x for x in os.listdir(args.rootPath) if not '.txt' in x])
    print('There are {} samples in total'.format(len(samples)))
    
    max_dimsN = [len(b) for b in priorN.values()]
    max_dimsP = [len(b) for b in priorP.values()]
    priorN = {torch.tensor(c): torch.tensor(b) for c, b in priorN.items()}
    priorP = {torch.tensor(c): torch.tensor(b) for c, b in priorP.items()}
    ph = {'0': topo.crip_wrapper, 'N': topo.trip_wrapper}
    
    results = np.ones((len(priorP), len(samples), 3)) * np.nan
    
    for i, sample in enumerate(samples):
        
        print('Processing sample = {}'.format(sample))
        mskPath = os.path.join(args.rootPath, sample, 'msk.nii')
        msk     = nib.load(mskPath)
        msk     = np.asarray(msk.dataobj).astype(int)
        msk[msk>3] = 3
        msk      = torch.tensor(msk[np.newaxis,:])
        nclasses = int(msk.max().numpy()) + 1
        one_hot  = F.one_hot(msk.long(), num_classes=nclasses)
        msk      = one_hot.permute(0, 4, 1, 2, 3).type(msk.type())

        if msk.shape[1] == 4:
            barcodes = topo.getBarcodes(msk[0,:,:,:,:], priorP, max_dimsP, ph, '0', False)
            for j in range(len(priorP)):
                a , c = np.unique(barcodes[j][:,0], return_counts=True)
                results[j,i,tuple(a.astype(int))] = c
        elif msk.shape[1] == 3:
            barcodes = topo.getBarcodes(msk[0,:,:,:,:], priorN, max_dimsN, ph, '0', False)
            for j in range(len(priorN)):
                a , c = np.unique(barcodes[j][:,0], return_counts=True)
                if j==2:j+=1 # to correctly select lv-myo
                results[j,i,tuple(a.astype(int))] = c
        else: raise ValueError("Wrong max in msk")
        
    with open(os.path.join(args.resPath, "{}.pickle".format(args.resName)), 'wb') as f:
        pickle.dump(results, f)
        
    plt.rcParams['figure.figsize'] = [15, 12]
    labels = ["lv", "myo", "mi", "lv-myo", "lv-mi", "myo-mi"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    x = np.arange(results.shape[1])
    for n,label in enumerate(labels):
        f, ax = plt.subplots(3,1)
        ax[0].scatter(x,results[n,:,0].T, c=colors[n], label=label)
        ax[1].scatter(x,results[n,:,1].T, c=colors[n], label=label)
        ax[2].scatter(x,results[n,:,2].T, c=colors[n], label=label)
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
        plt.show()
        plt.draw()
        fig.savefig(os.path.join(args.resPath, "{}_{}.png".format(args.resName, label)), dpi=100)
        plt.close()

if __name__ == '__main__':
    start = time.time()
    main(priorN, priorP)
    print("Total duration processing: {} s ".format(time.time()-start))