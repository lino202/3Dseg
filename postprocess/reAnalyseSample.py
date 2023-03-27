import argparse
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append(os.path.join('/'.join(sys.path[0].split("/")[:-1])))
import utils.topo as topo
import monai

prior_CINE_MnMs = {
    (1,):   (1, 0, 0),
    (2,):   (1, 1, 0), #Here maybe let open? (1,1,0) or close (1,0,0)
    (3,):   (1, 0, 0),
    (1, 2): (1, 0, 0),
    (1, 3): (2, 0, 0),
    (2, 3): (1, 1, 0)  #Here maybe let open? (1,1,0) or close (1,0,0)
}

prior_roi = {(1,):   (1, 0, 0)}

priorN_emidec = {
    (1,):   (1, 0, 0),
    (2,):   (1, 1, 0),
    (1, 2): (1, 0, 0),
}

priorP_emidec = {
    (1,):   (1, 0, 0),
    (2,):   (1, 1, 0),
    (3,):   (1, 0, 0),
    (1, 2): (1, 0, 0),
    (1, 3): (1, 0, 0),
    (2, 3): (1, 1, 0)
}

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--predPath', required=True, type=str)
    parser.add_argument('--mskPath', required=True, type=str)
    parser.add_argument('--priorName', required=True, type=str)
    parser.add_argument('--nclasses', required=True, type=int)
    args = parser.parse_args()
    
    samples = os.listdir(args.predPath)
    prior = globals()[args.priorName]
    
    for i, sample in enumerate(samples):
        print("Sample: {} {}".format(i, sample))
    
        msk  = nib.load(os.path.join(args.mskPath, sample, "msk.nii"))
        msk = np.asarray(msk.dataobj)
        print(np.unique(msk))
        msk = torch.tensor(msk[np.newaxis,:])
        one_hot = F.one_hot(msk.long(), num_classes=args.nclasses)
        msk    = one_hot.permute(0, 4, 1, 2, 3).type(msk.type())
        
        pred = nib.load(os.path.join(args.predPath, sample, "pred.nii"))
        pred = np.asarray(pred.dataobj).astype(int)
        print(np.unique(pred))
        pred = torch.tensor(pred[np.newaxis,:])
        one_hot = F.one_hot(pred.long(), num_classes=args.nclasses)
        pred    = one_hot.permute(0, 4, 1, 2, 3).type(pred.type())
        
        gdsc  = monai.metrics.compute_generalized_dice(torch.permute(pred, (1,0,2,3,4)), torch.permute(msk, (1,0,2,3,4)), include_background=True)
        hd    = monai.metrics.compute_hausdorff_distance(pred, msk, include_background=True)
        be    = topo.BEmetric(pred[0,:,:,:], msk[0,:,:,:], prior, parallel=False)
        if be == 0.: 
            ts = 1 
        else: 
            ts = 0
        
        print(gdsc)
        print(hd)
        print(be)
        print(ts)

if __name__ == '__main__':
    main()