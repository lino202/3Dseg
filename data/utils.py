import torch
import numpy as np


def customToTensor(arr, arrType, isgan):
    '''Here we applied our tranforms for prepocessing
    We adjust some built in functions as they were not thought for 3D arrays, 
    e.g. there's a rotation in toTensor
    The resulting img is an isotropic tensor that might be rotate around LA and 
    normalize in the range of [-1,1] with an extra C=channels dimension 
    The resulting msk is just a tensor with no normalization and no added dimension as C=1
    is enough. The batch size is added in by the dataloader.'''

    if arrType != 'msk': 
        arr = unitNorm(arr)
        tensor = torch.as_tensor(arr, dtype=torch.float32).contiguous() #default net parameters init is in torch.Float (32bytes) no torch.Double (64bytes)
        tensor = normalize(tensor, (0.5,), (0.5,))
        tensor = addDim(tensor)
    else:
        if isgan: #get msk in the range [-1,1] for using as net input (GAN and pred GAN)
            arr = unitNorm(arr) #between [0,1]
            arr = (arr * 2) - 1
            tensor = torch.as_tensor(arr, dtype=torch.float32).contiguous()
            tensor = addDim(tensor)
        else: #Use msk as ground truth
            tensor = torch.as_tensor(arr, dtype=torch.int64).contiguous()
    return tensor

def unitNorm(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def normalize(tensor, mean, std):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)

def addDim(tensor):
    return tensor.unsqueeze(0)