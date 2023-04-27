

import numpy as np
import os
from utilsROI import getROICINE, getROIMsk

import torchio as tio    
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import SpatialTransform, ElasticDeformation
from torchio.typing import TypeTripletInt
from torchio import Subject
from typing import Tuple
from typing import Union
from torchio.utils import to_tuple
from torchio.transforms.augmentation.spatial.random_elastic_deformation import _parse_num_control_points, _parse_max_displacement
import torch

class myRandomElasticDeformation(RandomTransform, SpatialTransform):
    def __init__(
        self,
        num_control_points: Union[int, Tuple[int, int, int]] = 7,
        max_displacement: Union[float, Tuple[float, float, float]] = 7.5,
        locked_borders: int = 2,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        **kwargs
    ):
        super().__init__(**kwargs)
        self._bspline_transformation = None
        self.num_control_points = to_tuple(num_control_points, length=3)
        _parse_num_control_points(self.num_control_points)  # type: ignore[arg-type]  # noqa: E501
        self.max_displacement = to_tuple(max_displacement, length=3)
        _parse_max_displacement(self.max_displacement)  # type: ignore[arg-type]  # noqa: E501
        self.num_locked_borders = locked_borders
        if locked_borders not in (0, 1, 2):
            raise ValueError('locked_borders must be 0, 1, or 2')
        if locked_borders == 2 and 4 in self.num_control_points:
            message = (
                'Setting locked_borders to 2 and using less than 5 control'
                'points results in an identity transform. Lock fewer borders'
                ' or use more control points.'
            )
            raise ValueError(message)
        self.image_interpolation = self.parse_interpolation(
            image_interpolation,
        )
        self.label_interpolation = self.parse_interpolation(
            label_interpolation,
        )
        
        #Init Elastic Transformation
        control_points = self.get_params(
            self.num_control_points,  # type: ignore[arg-type]
            self.max_displacement,  # type: ignore[arg-type]
            self.num_locked_borders,
        )
        
        arguments = {
            'control_points': control_points,
            'max_displacement': self.max_displacement,
            'image_interpolation': self.image_interpolation,
            'label_interpolation': self.label_interpolation,
        }

        self.transform = ElasticDeformation(**self.add_include_exclude(arguments))

    @staticmethod
    def get_params(
        num_control_points: TypeTripletInt,
        max_displacement: Tuple[float, float, float],
        num_locked_borders: int,
    ) -> np.ndarray:
        grid_shape = num_control_points
        num_dimensions = 3
        coarse_field = torch.rand(*grid_shape, num_dimensions)  # [0, 1)
        coarse_field -= 0.5  # [-0.5, 0.5)
        coarse_field *= 2  # [-1, 1]
        for dimension in range(3):
            # [-max_displacement, max_displacement)
            coarse_field[..., dimension] *= max_displacement[dimension]

        # Set displacement to 0 at the borders
        for i in range(num_locked_borders):
            coarse_field[i, :] = 0
            coarse_field[-1 - i, :] = 0
            coarse_field[:, i] = 0
            coarse_field[:, -1 - i] = 0

        return coarse_field.numpy()

    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_spatial_shape()
        transformed = self.transform(subject)
        assert isinstance(transformed, Subject)
        return transformed
   

def crop(arr, roi):
    xmin, xmax = int(roi[0] - roi[2]), int(roi[0] + roi[2])
    ymin, ymax = int(roi[1] - roi[3]), int(roi[1] + roi[3])
    if xmin < 0: xmin=0
    if ymin < 0: ymin=0 
    if xmax > arr.shape[0]-1: xmax=arr.shape[0]-1
    if ymax > arr.shape[1]-1: ymax=arr.shape[1]-1
    arrCropped = arr[xmin:xmax,ymin:ymax,:]
    return arrCropped
    
    
def cropSubjectPred(subject, pred):
    ''' This code takes a subject from torchio and perform cropping on both with a cardiac ROI''' 
    imgArr  = subject.img.data.numpy()
    mskArr  = subject.msk.data.numpy()
    predArr = pred.data.numpy()[0,:,:,:]
    roi_min, roi_max = getROIMsk(predArr)
    
    cropMask = np.zeros(subject.img.shape[-3:])
    cropMask[roi_min[0]:roi_max[0], roi_min[1]:roi_max[1], roi_min[2]:roi_max[2]] = 1

    trans = tio.CropOrPad((roi_max[0]-roi_min[0], roi_max[1]-roi_min[1], roi_max[2]-roi_min[2]), mask_name='cropMask')
    subject['cropMask'] = tio.LabelMap(tensor=cropMask[np.newaxis,:],affine=subject.msk.affine)
    transSubject = trans(subject)

    return transSubject


def cropSubjectCINE(subject, edes):
    ''' This code takes a subject from torchio and perform cropping on both with a cardiac ROI''' 
    imgArr = subject.img.data.numpy()
    mskArr = subject.msk.data.numpy()
    roi = getROI(imgArr)
    
    #Adjuts roi with mask
    mskArrCropED = crop(mskArr[edes[0],:,:,:], roi)
    while not np.count_nonzero(mskArr[edes[0],:,:,:]) == np.count_nonzero(mskArrCropED):
        print ("Changing ROI, control this sample with slicer")
        roi = roi + np.array([0,0,10,10])
        mskArrCropED = crop(mskArr[edes[0],:,:,:], roi)    
    roi = roi + np.array([0,0,20,20])

    #Get mask crop for torchio function and targets
    #For now we are cropping x-y axis and not z
    #Maybe the roi (z axis) can be extracted from movement in the coronal plane
    cropMask = np.zeros(subject.img.shape[-3:])
    xmin, xmax = int(roi[0] - roi[2]), int(roi[0] + roi[2])
    ymin, ymax = int(roi[1] - roi[3]), int(roi[1] + roi[3])
    if xmin < 0: xmin=0
    if ymin < 0: ymin=0 
    if xmax > subject.img.shape[1]-1: xmax=subject.img.shape[1]-1
    if ymax > subject.img.shape[2]-1: ymax=subject.img.shape[2]-1
    cropMask[xmin:xmax,ymin:ymax,:] = 1
    
    trans = tio.CropOrPad((xmax-xmin, ymax-ymin, subject.img.shape[3]), mask_name='cropMask')
    subject['cropMask'] = tio.LabelMap(tensor=cropMask[np.newaxis,:],affine=subject.msk.affine)
    transSubject = trans(subject)

    return transSubject

def getEXFromMask(msk, sample):
    '''This finds the EX indexes for 4D ground truth data
    the time should be first axis, this is compliant with torchio''' 
    edes = np.zeros(msk.shape[0])
    for t in range(msk.shape[0]):
        if np.max(msk[t,:,:,:]) != 0: 
            edes[t] = 1    
    if np.sum(edes) == 2:
        edesIdx = np.nonzero(edes)[0]
        if np.count_nonzero(msk[edesIdx[0],:,:,:]) >= np.count_nonzero(msk[edesIdx[1],:,:,:]):
            return edesIdx[0], edesIdx[1]
        else:
            return edesIdx[1], edesIdx[0]
    else:
        raise ValueError("Control the mask of sample {} in time, different than 2 (ED and ES)".format(sample))

        

# def getBiV(arr, dilaIteractions, RVBPClass=3, LVMClass=2, kernel=(2,2)):
#     '''This function dilate the RVBP for generating a BiV segmentation''' 
#     kernel = scipy.ndimage.generate_binary_structure(kernel[0],kernel[1])
#     rows, cols, slices = arr.shape
#     arrBiVent = np.zeros(arr.shape)
#     for s in range(slices):
#         mskTmp = copy.deepcopy(arr[:,:,s])
#         if np.max(mskTmp) != 0:
#             mskTmp[mskTmp!=RVBPClass] = 0
#             mskDila = scipy.ndimage.binary_dilation(mskTmp, structure=kernel, iterations=dilaIteractions).astype(mskTmp.dtype)
#             RVmyo=mskDila*3-mskTmp
#             LVmyo = copy.deepcopy(arr[:,:,s])
#             LVmyo[LVmyo!=LVMClass] = 0
#             newMsk = LVmyo + RVmyo
#             newMsk[newMsk!=0]=1
#             arrBiVent[:,:,s] = newMsk
#     return arrBiVent
