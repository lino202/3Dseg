''' Code from https://github.com/nick-byrne/topological-losses based on 
N. Byrne, J. R. Clough, I. Valverde, G. Montana and A. P. King, 
"A persistent homology-based topological loss for CNN-based multi-class segmentation of CMR," 
in IEEE Transactions on Medical Imaging, 2022, doi: 10.1109/TMI.2022.3203309.'''

from multiprocessing import Pool
import cripser as crip
import tcripser as trip
import numpy as np
import torch
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import os

def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)

def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)

def get_roi(X, thresh=0.01):
    true_points = torch.nonzero(X >= thresh)
    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]
    roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
    return roi

def getBarcodes(tensor, prior, max_dims, ph, construction, parallel=True):
    #tensor must be ont-hot encoded

    # Build class/combination-wise (c-wise) image tensor for prior
    tmp = []
    for c in prior.keys():
        if c.dim()== 1:
            tmp.append(tensor[c].sum(0))
        else:
            raise ValueError("Wrong dim")
    combos = torch.stack(tmp)

    # Invert probababilistic fields for consistency with cubical ripser sub-level set persistence
    combos = 1 - combos

    # Get barcodes using cripser in parallel without autograd            
    combos_arr = combos.detach().cpu().numpy().astype(np.float64)
    if parallel:
        with torch.no_grad():
            with Pool(len(prior)) as p:
                bcodes_arr = p.starmap(ph[construction], zip(combos_arr, max_dims))
    else:
        with torch.no_grad():
            bcodes_arr = [ph[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]
            
    return bcodes_arr


def BEmetric(pred, msk, prior, parallel, maxdim=2, construction='0'):
    '''Performs Get Betti error as defined in the Byrne et al, paper.
    
    Arguments:
        REQUIRED
        pred         - PyTorch tensor with shape [1, number of classes, H,W,[D]]
        msk          - PyTorch tensor with shape [1, number of classes, H,W,[D]]
        prior        - Topological prior as dictionary:
                        keys are tuples specifying the channel(s) of inputs
                        values are tuples specifying the desired Betti numbers
 
        OPTIONAL [default]
        maxdim       - max dimension for PH calculation [2] as we are working with 3D samples
        construction - Either '0' (4 (2D) or 6 (3D) connectivity) or 'N' (8 (2D) or 26 (3D) connectivity) ['0']'''
    
    #Check for shape
    assert(pred.shape==msk.shape)
     
    # Inspect prior and convert to tensor
    max_dims = [len(b) for b in prior.values()]
    prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    
    ph = {'0': crip_wrapper, 'N': trip_wrapper}
    
    #Get barcodes
    predBarcodes = getBarcodes(pred, prior, max_dims, ph, construction, parallel)
    mskBarcodes  = getBarcodes(msk, prior, max_dims, ph, construction, parallel) 
    
    be = np.zeros(len(prior))
    for i in range(len(prior)):
        be[i] = abs(predBarcodes[i].shape[0] - mskBarcodes[i].shape[0] )
    
    return np.sum(be)    


def checkIndexing(indexes, shape):
    shapeUpLimit = np.array(shape)[np.newaxis,:] - 1
    shapeDownLimit = np.array((0,0,0))[np.newaxis,:]
    idxsOut    = (indexes > shapeUpLimit).nonzero()
    indexes[idxsOut] = shapeUpLimit[0,idxsOut[1]]
    idxsOut    = (indexes < shapeDownLimit).nonzero()
    indexes[idxsOut] = shapeDownLimit[0,idxsOut[1]]


def get_differentiable_barcode(tensor, barcode, shape):
    '''Makes the barcode returned by CubicalRipser differentiable using PyTorch.
    Note that the critical points of the CubicalRipser filtration reveal changes in sub-level set topology.
    
    Arguments:
        REQUIRED
        tensor  - PyTorch tensor w.r.t. which the barcode must be differentiable
        barcode - Barcode returned by using CubicalRipser to compute the PH of tensor.numpy() 
        shape   - The shape of the tensor that was obtained from combo_arr in order to avoid bad indexing due to ph N constructor overpassing tensor size
    '''
    # Identify connected component of ininite persistence (the essential feature)
    inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max]
    fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]
    
    # Get birth of infinite feature
    # with construction N sometimes the tensor is indexed out of range which causes 
    # IndexKernel.cu:91: block: [0,0,0], thread: [0,0,0] Assertion `index >= -sizes[i] && index < sizes[i] && "index out of bounds"` failed.
    # This is because the barcode gives indexes (xyz birth or death) out of range so we checked and adjusted to the boundaries. I saw this for the 
    # inf entity but I applied it to the others as well just in case
    indexes = inf[:, 3:3+tensor.ndim].astype(np.int64)
    checkIndexing(indexes, shape)
    inf_birth = tensor[tuple(indexes.T)]
    
    # Calculate lifetimes of finite features
    indexes = fin[:, 3:3+tensor.ndim].astype(np.int64)
    checkIndexing(indexes, shape)
    births = tensor[tuple(indexes.T)]
    indexes = fin[:, 6:6+tensor.ndim].astype(np.int64)
    checkIndexing(indexes, shape)
    deaths = tensor[tuple(indexes.T)]
    delta_p = (deaths - births)
    
    # Split finite features by dimension
    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    
    # Sort finite features by persistence
    delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    
    return inf_birth, delta_p

def multi_class_topological_post_processing(
    inputs, model, prior,
    lr, mse_lambda,
    opt=torch.optim.Adam, num_its=100, 
    construction='0', thresh=None, parallel=True, saveCombosPath=None):
    '''Performs topological post-processing.
    
    Arguments:
        REQUIRED
        inputs       - PyTorch tensor - [1, number of classes] + [spatial dimensions (2D or 3D)]
        model        - Pre-trained CNN as PyTorch module (without final activation)
        prior        - Topological prior as dictionary:
                       keys are tuples specifying the channel(s) of inputs
                       values are tuples specifying the desired Betti numbers
        lr           - Learning rate for SGD optimiser
        mse_lambda   - Weighting for similarity constraint
        
        OPTIONAL [default]
        opt          - PyTorch optimiser [torch.optim.Adam]
        num_its      - Iterable of number iterations(s) to run for each scale [100]
        construction - Either '0' (4 (2D) or 6 (3D) connectivity) or 'N' (8 (2D) or 26 (3D) connectivity) ['0']
        thresh       - Threshold at which to define the foreground ROI for topological post-processing
    '''
    # Get image properties
    spatial_xyz = list(inputs.shape[2:])
    
    # Get working device
    device = inputs.device
    
    # Get raw prediction
    model.eval()
    with torch.no_grad():
        pred_unet = torch.softmax(model(inputs), 1).detach().squeeze()
        
    # If appropriate, choose ROI for topological consideration
    if thresh:
        roi = get_roi(pred_unet[1:].sum(0).squeeze(), thresh)
    else:
        roi = [slice(None, None)] + [slice(None, None) for dim in range(len(spatial_xyz))]
    
    # Initialise topological model and optimiser
    model_topo = copy.deepcopy(model)
    model_topo.train()                              #Error in original?
    optimiser = opt(model_topo.parameters(), lr=lr)
    
    # Inspect prior and convert to tensor
    max_dims = [len(b) for b in prior.values()]
    prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    
    # Set mode of cubical complex construction
    ph = {'0': crip_wrapper, 'N': trip_wrapper}

    for it in range(num_its):

        # Reset gradients
        optimiser.zero_grad()

        # Get current prediction
        outputs = torch.softmax(model_topo(inputs), 1).squeeze()
        outputs_roi = outputs[roi]

        # Build class/combination-wise (c-wise) image tensor for prior
        # Here we adjust this to do not have the c.T warning
        # as all the c or prior keys should be 1D then is not reason to do c.T
        # pytorch raises a warning
        tmp = []
        for c in prior.keys():
            if c.dim()== 1:
                tmp.append(outputs_roi[c].sum(0))
            else:
                raise ValueError("Wrong dim")
        combos = torch.stack(tmp)

        if it%25==0 or it==99:
            if saveCombosPath:
                combosArr = combos.cpu().detach().numpy()
                nImgs = combosArr.shape[0]+1
                shown_slice = int(combosArr.shape[3]/2)
                ncols=3
                nrows = int(np.ceil(nImgs/ncols))
                _, a = plt.subplots(nrows, ncols)
                if a.ndim == 1 : a = a[np.newaxis,:]
                j=0
                i=0
                for nImg in range(nImgs-1):
                    a[j,i].imshow(combosArr[nImg,:,:,shown_slice], interpolation='none')
                    a[j,i].axis('off')
                    i+=1
                    if i % ncols == 0: 
                        j += 1
                        i = 0
                a[j,i].imshow(inputs[0][roi].cpu().numpy()[0,:,:,shown_slice], interpolation='none')
                plt.savefig(os.path.join(saveCombosPath, "it_{}.png".format(it)), dpi=300)
                plt.close()

        # Invert probababilistic fields for consistency with cripser sub-level set persistence
        combos = 1 - combos

        # Get barcodes using cripser in parallel without autograd            
        combos_arr = combos.detach().cpu().numpy().astype(np.float64)
        combo_shape = combos_arr.shape[1:]
        if parallel:
            with torch.no_grad():
                with Pool(len(prior)) as p:
                    bcodes_arr = p.starmap(ph[construction], zip(combos_arr, max_dims))
        else:
            with torch.no_grad():
                bcodes_arr = [ph[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]

        # Get differentiable barcodes using autograd
        max_features = max([bcode_arr.shape[0] for bcode_arr in bcodes_arr])
        bcodes = torch.zeros([len(prior), max(max_dims), max_features], requires_grad=False, device=device)
        for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
            _, fin = get_differentiable_barcode(combo, bcode, combo_shape)
            for dim in range(len(spatial_xyz)):
                bcodes[c, dim, :len(fin[dim])] = fin[dim]

        # Select features for the construction of the topological loss
        stacked_prior = torch.stack(list(prior.values()))
        stacked_prior.T[0] -= 1 # Since fundamental 0D component has infinite persistence
        matching = torch.zeros_like(bcodes, dtype=torch.uint8).detach()

        #Do not touch the ones I am not certain about the topo
        #matching is not bool anymore, now it is: 
        #0 = not matched/incorrect
        #1 = correct
        #2 = I do not know (so neither correct neither incorrect in loss)
        bcodes_arr = bcodes.detach().cpu().numpy()
        for c, combo in enumerate(stacked_prior):
            for dim in range(len(combo)):
                if stacked_prior[c, dim] >= 0: # If user put a certain topology
                    matching[c, dim, slice(None, stacked_prior[c, dim])] = 1
                else: # If user put -1 (dubious topo)
                    nTopos = np.count_nonzero(bcodes_arr[c,dim,:])
                    matching[c, dim, slice(None, nTopos)] = 2
        

        # Find total persistence of features which match (A) / violate (Z) the prior
        A = (1 - bcodes[matching==1]).sum()
        Z = bcodes[matching==0].sum()

        # Get similarity constraint
        mse = F.mse_loss(outputs, pred_unet)

        # Optimisation
        loss = A + Z + mse_lambda * mse
        loss.backward()
        optimiser.step()

    return model_topo




