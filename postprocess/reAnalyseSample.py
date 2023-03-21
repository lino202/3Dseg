import argparse
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
# import sys
import monai


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--predPath', required=True, type=str)
    parser.add_argument('--mskPath', required=True, type=str)
    parser.add_argument('--fileIdx', required=True, type=int)
    args = parser.parse_args()
    
    samples = os.listdir(args.predPath) #[args.fileIdx]
    
    for i, sample in enumerate(samples):
        print("Sample: {} {}".format(i, sample))
    
        msk  = nib.load(os.path.join(args.mskPath, sample, "msk.nii"))
        msk = np.asarray(msk.dataobj)
        msk = torch.tensor(msk[np.newaxis,:])
        nclasses = int(msk.max().numpy()) + 1
        one_hot = F.one_hot(msk.long(), num_classes=nclasses)
        msk    = one_hot.permute(0, 4, 1, 2, 3).type(msk.type())
        
        pred = nib.load(os.path.join(args.predPath, sample, "pred.nii"))
        pred = np.asarray(pred.dataobj).astype(int)
        pred = torch.tensor(pred[np.newaxis,:])
        one_hot = F.one_hot(pred.long(), num_classes=nclasses)
        pred    = one_hot.permute(0, 4, 1, 2, 3).type(pred.type())
        
        gdsc  = monai.metrics.compute_generalized_dice(torch.permute(pred, (1,0,2,3,4)), torch.permute(msk, (1,0,2,3,4)), include_background=True)
        hd    = monai.metrics.compute_hausdorff_distance(pred, msk, include_background=True)
        # be    = topo.BEmetric(pred[0,:,:,:], msk[0,:,:,:], prior_CINE_MnMs)
        # if be == 0.: ts[i] = 1
        
        print(gdsc)
        print(hd)

if __name__ == '__main__':
    main()