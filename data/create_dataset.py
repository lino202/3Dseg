import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import grid_sample
import os
import nibabel as nib
import numpy as np
import random

def get_transform(transformParams, arrType, times):
    
    transform_list = []
    size = [transformParams["load_size_h"], transformParams["load_size_w"], transformParams["load_size_d"]]
    transform_list.append(transforms.Lambda(lambda arr: resize3D(arr, size, arrType)))
    transform_list.append(transforms.Lambda(lambda arr: unitNorm(arr)))
    if not transformParams["no_hor_flip"]: transform_list.append(transforms.Lambda(lambda arr: rotZPlane(arr, times)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    transform_list.append(transforms.Lambda(lambda tensor: addChannel(tensor)))

    return transforms.Compose(transform_list)

def create(opt, phase):

    transform = {"load_size_d": opt.load_size_d,
            "load_size_h": opt.load_size_h,
            "load_size_w": opt.load_size_w,
            "no_hor_flip": opt.no_hor_flip}

    if phase == "val":
        transform["no_hor_flip"] = True


    dataset = Dataset3D(opt.root_path, phase, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)

    return dataloader

def resize3D(arr, size, arrType):    
    x = torch.Tensor( np.reshape(arr, (1,1,arr.shape[0],arr.shape[1],arr.shape[2])))
    h = torch.linspace(-1, 1, size[0])
    w = torch.linspace(-1, 1, size[1])
    d = torch.linspace(-1, 1, size[2])
    meshz, meshy, meshx = torch.meshgrid((h, w, d))
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0) # add batch dim
    tensor = grid_sample(x, grid, align_corners=True)
    arr = tensor[0,0,:,:,:].numpy()

    #As this does an interpolation the mask should still have only 0's or 1's
    if arrType == "msk": 
        arr[arr<0.5] = 0.
        arr[arr>=0.5] = 1.
    return arr

def unitNorm(arr):
    return (arr -np.min(arr)) / (np.max(arr)-np.min(arr))

def addChannel(tensor):
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
        self.img = nib.load( os.path.join(self.files[index], "img.nii") ).get_fdata()
        self.msk = nib.load( os.path.join(self.files[index], "msk.nii") ).get_fdata()
        
        times = random.randint(0,3) #for flipping
        imgTrans = get_transform(self.transform, "img", times)
        mskTrans = get_transform(self.transform, "msk", times)

        self.img = imgTrans(self.img)
        self.msk = mskTrans(self.msk)

        return {"img": self.img, "msk": self.msk, "path": self.files[index]}


    def __len__(self):
        return self.nSamples



