

import numpy as np
import os
from utilsROI import getROI
import torchio as tio       

def crop(arr, roi):
    xmin, xmax = int(roi[0] - roi[2]), int(roi[0] + roi[2])
    ymin, ymax = int(roi[1] - roi[3]), int(roi[1] + roi[3])
    if xmin < 0: xmin=0
    if ymin < 0: ymin=0 
    if xmax > arr.shape[0]-1: xmax=arr.shape[0]-1
    if ymax > arr.shape[1]-1: ymax=arr.shape[1]-1
    arrCropped = arr[xmin:xmax,ymin:ymax,:]
    return arrCropped


def cropSubject(subject, edes):
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
