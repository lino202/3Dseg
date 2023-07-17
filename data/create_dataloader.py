from torch.utils.data import Dataset, DataLoader
import os
import torchio as tio
from .utils import customToTensor

def create(opt, phase):
    ''' We create the dataset and dataloader, relying on the phase
    Four phases are available:
        train = img and msk are generated with shuffle and dataaug (if desired)
        val   = the same as train wihout dataaug
        test  = the same as val without shuffle
        pred  = the same as test and we do not have the ground truth so no msk is generated'''

    #Dataset
    if phase == "pred":
        # dataset = Dataset3DPred(opt.root_path, opt.gan)
        raise NotImplementedError
    elif phase == "test" or phase == "val":
        dataset = Dataset3D(opt.root_path, phase, opt.gan, False)
    elif phase == "train":
        dataset = Dataset3D(opt.root_path, phase, opt.gan, opt.dataaug)
    else: raise NotImplementedError

    #Dataloader
    if phase == "train" or phase == "val":
        dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
    elif phase == "test" or phase == "pred":
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    else:
        raise ValueError("Wrong Phase: {}".format(phase))
    return dataloader


class Dataset3D(Dataset):
    
    def __init__(self, rootPath, phase, isgan, dataaug):
        self.dataPath  = os.path.join(rootPath, phase)
        self.files     = [os.path.join(self.dataPath, x) for x in os.listdir(self.dataPath)]
        self.nSamples  = len(self.files)
        self.isgan     = isgan
        self.dataaug   = dataaug

        if not self.isgan:
            self.transform = tio.Compose([
                tio.RandomAffine(degrees=(0.,0.,180.),p=0.4),
                tio.RandomFlip(axes="L", flip_probability=1.0, p=0.5),
                tio.RandomFlip(axes="A", flip_probability=1.0, p=0.5),
                tio.RandomBlur(p=0.4),                    # blur 25% of times
                tio.RandomNoise(p=0.4),                   # Gaussian noise 25% of times  
                tio.RandomBiasField(p=0.4),                # magnetic field inhomogeneity 30% of times
                tio.OneOf({                                # either
                    tio.RandomMotion(): 2,                 # random motion artifact
                    tio.RandomSpike(intensity=(0.1,0.6)): 2, # or spikes
                    tio.RandomGhosting(): 2,                 # or ghosts
                }, p=0.5),                                   # applied to 50% of images
            ])
        else:
            self.transform = tio.Compose([
                tio.RandomAffine(degrees=(0.,0.,180.),p=0.4),
                tio.RandomFlip(axes="L", flip_probability=1.0, p=0.5),
                tio.RandomFlip(axes="A", flip_probability=1.0, p=0.5),
            ])

 

    def __getitem__(self, index):
        imgPath = os.path.join(self.files[index], "img.nii")
        mskPath = os.path.join(self.files[index], "msk.nii")
        subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        subject.check_consistent_attribute('spacing') 
        subject.check_consistent_attribute('affine')
        subject.check_consistent_attribute('shape')

        if self.dataaug:
            #We impose an affine to be the identity in order to perform the affine rotations correctly
            subject = tio.Subject(img=tio.ScalarImage(tensor=subject.img.data), 
                                  msk=tio.LabelMap(tensor=subject.msk.data))
            subject = self.transform(subject)
        self.img = subject.img.data.numpy()[0]
        self.msk = subject.msk.data.numpy()[0]
        self.affine = subject.img.affine

        #The final transformation takes into account if gan or not
        self.img = customToTensor(self.img, "img", self.isgan)
        self.msk = customToTensor(self.msk, "msk", self.isgan)

        return {"img": self.img, "msk": self.msk, "path": self.files[index], "affine": self.affine}


    def __len__(self):
        return self.nSamples
    

# class Dataset3DPred(Dataset):
#     ''' Prediction is use when only the net input is available and we are not going to 
#     use the ground truth, this input can be the msk or img depending if we use the GANs
#     generator or not (normal approach)'''
    
#     def __init__(self, rootPath, isgan):
#         self.dataPath  = os.path.join(rootPath)
#         self.files     = [os.path.join(self.dataPath, x) for x in os.listdir(self.dataPath)]
#         self.nSamples  = len(self.files)
#         self.isgan     = isgan

#     def __getitem__(self, index):
        
#         if not self.isgan:
#             self.img=tio.ScalarImage(os.path.join(self.files[index], "img.nii"))
#         else:
#             self.img=tio.LabelMap(os.path.join(self.files[index], "msk.nii"))
        
#         self.affine = self.img.affine
#         self.img    = self.img.data.numpy()[0]
        
#         if not self.isgan:
#             self.img = customToTensor(self.img, "img", self.isgan)
#         else:
#             self.img = customToTensor(self.img, "msk", self.isgan)

#         return {"img": self.img, "path": self.files[index], "affine": self.affine}

#     def __len__(self):
#         return self.nSamples