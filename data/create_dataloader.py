import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import grid_sample
import os
import nibabel as nib
import numpy as np
import random

def get_transform(transformParams, arrType, times):
    #Here we applied our tranforms for prepocessing
    #We adjust some built in functions as they were not thought for 3D arrays, 
    #e.g. there's a rotation in toTensor
    #The resulting img is an isotropic tensor that might be rotate around LA and 
    #normalize in the range of [-1,1] with an extra C=channels dimension 
    #The resulting msk is just a tensor with no normalization and no added dimension as C=1
    #is enough. The batch size is added in by the dataloader.
    transform_list = []
    
    size = (transformParams["load_size_h"], transformParams["load_size_w"], transformParams["load_size_d"])
    transform_list.append(transforms.Lambda(lambda arr: resize3D(arr, size, arrType)))
    if not transformParams["no_hor_flip"]: transform_list.append(transforms.Lambda(lambda arr: rotZPlane(arr, times)))
    transform_list.append(transforms.Lambda(lambda arr: toTensor(arr, arrType)))
    # if arrType != 'msk': transform_list.append(transforms.Lambda(lambda tensor: normalize(tensor, (0.5,), (0.5,))))
    # transform_list.append(transforms.Lambda(lambda tensor: addDim(tensor)))

    return transforms.Compose(transform_list)

def create(opt, phase):

    transform = {"load_size_d": opt.load_size_d,
            "load_size_h": opt.load_size_h,
            "load_size_w": opt.load_size_w,
            "no_hor_flip": opt.no_hor_flip}

    if phase == "val" or phase == "test":
        transform["no_hor_flip"] = True

    dataset = Dataset3D(opt.root_path, phase, transform)
    if phase != "test":
        dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False)
    
    return dataloader

def resize3D(arr, size, arrType):    
    if arr.shape != size:
        x = torch.Tensor( np.reshape(arr, (1,1,arr.shape[0],arr.shape[1],arr.shape[2])))
        h = torch.linspace(-1, 1, size[0])
        w = torch.linspace(-1, 1, size[1])
        d = torch.linspace(-1, 1, size[2])
        meshz, meshy, meshx = torch.meshgrid((h, w, d), indexing='ij')
        grid = torch.stack((meshx, meshy, meshz), 3)
        grid = grid.unsqueeze(0) # add batch dim
        tensor = grid_sample(x, grid, align_corners=True)
        arr = tensor[0,0,:,:,:].numpy()

        #As this does an interpolation the mask should still have only 0's or 1's
        #This should be improved for multiclass segmentation, so for only have to 
        #be use with binary segmentation
        if arrType == "msk": 
            arr[arr<0.5] = 0.
            arr[arr>=0.5] = 1.

    return arr

def toTensor(arr, arrType):
    if arrType != 'msk':
        arr = unitNorm(arr)
        tensor = torch.as_tensor(arr, dtype=torch.float32).contiguous() #default net parameters init is in torch.Float (32bytes) no torch.Double (64bytes)
        tensor = normalize(tensor, (0.5,), (0.5,))
        tensor = addDim(tensor)
    else: 
        tensor = torch.as_tensor(arr, dtype=torch.int64).contiguous() #dtype=torch.int64
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

def rotZPlane(arr, times):
    return np.rot90(arr, k=times, axes=(0,1)).copy()

class Dataset3D(Dataset):
    
    def __init__(self, rootPath, phase, transform):
        self.dataPath = os.path.join(rootPath, phase)
        self.files = [os.path.join(self.dataPath, x) for x in os.listdir(self.dataPath)]
        self.nSamples = len(self.files)
        self.transform = transform

    def __getitem__(self, index):
        self.img    = nib.load( os.path.join(self.files[index], "img.nii") )
        self.affine = self.img.affine
        self.img    = np.asarray(self.img.dataobj).astype(float)
        self.msk    = nib.load( os.path.join(self.files[index], "msk.nii") )
        self.msk    = np.asarray(self.msk.dataobj)
        
        times = random.randint(0,3) #for flipping
        imgTrans = get_transform(self.transform, "img", times)
        mskTrans = get_transform(self.transform, "msk", times)
        
        # import matplotlib.pyplot as plt
        # f, ax = plt.subplots(1,2)
        # ax[0].imshow(np.squeeze(self.msk[:,:,50]))
        # ax[1].imshow(np.squeeze(self.img[:,:,50]))
        # plt.figure(), plt.imshow(np.squeeze(self.img.numpy()[:,:,:,50])),plt.show()
        # plt.show()
        
        self.img = imgTrans(self.img)
        self.msk = mskTrans(self.msk)

        return {"img": self.img, "msk": self.msk, "path": self.files[index], "affine": self.affine}


    def __len__(self):
        return self.nSamples



