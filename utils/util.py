import numpy as np
import torch 
import os 
from PIL import Image

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def getBaseMidApexImgs(tensor, name):
    ''' Here we collect three images per volume from the base, apex and middle regions.
    The tensor-image outcome should be in the form of B,C=1,W,H where B=batch size, C=channels, W=width, H=height.
    Also the values range should cover the interval [-1,1].
    The ulterior function tensor2im take care of normalizing the results to [0,255] and use the first sample in the batch.
    
    Different Inputs shapes and ranges (B=batch size, C=channels, W=width, H=height, D=Depth)
    Msk: shape B,W,H,D   ranges [0,1] Binary or [0,N] Multi-class
    Img: shape B,1,W,H,D ranges [-1,1] (only single-channel volumes are used)
    Pred:shape B,C,W,H,D ranges [-1,1] and C=1 (binary) or C=N (multi-class)
    '''
    #Here we detach the gradients calculation for the tensor
    #The resulting tensor should be a B,1,W,H,D tensor with range [-1,1]
    tensor  = tensor.detach()
    if name == 'pred':
        #Predict can have C=1 (Binary) or C=Number of classes (Multi-class)
        if tensor.shape[1] > 1: #Multi-class
            tensor = torch.argmax(torch.softmax(tensor,1),1)
            tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 2 - 1
            tensor = tensor[:,np.newaxis,:,:,:]
    elif name == 'img':
        pass #not need to change anything
    elif name == 'msk':
        tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 2 - 1
        tensor = tensor[:,np.newaxis,:,:,:]
    else: raise ValueError("Wrong sample tag")
    
    # We select the three slices to plot
    # The naming of base and apex might be inverted, this relies on the orientation in depth of the images
    # if this is from base-apex or from apex-base (as this only influence the naming in the plotting this is irrelevant)
    nSlices = tensor.size()[-1]
    baseIdx = nSlices - 3
    apexIdx = 2 
    midIdx = int((baseIdx - apexIdx)/2)
    if midIdx < 0: raise ValueError("Wrong midIdx") 
    
    #Generate final image dict with final per image shape B,1,W,H
    imgsDict = {name + "_base": tensor[:,:,:,:,baseIdx],
                name + "_mid":  tensor[:,:,:,:,midIdx],
                name + "_apex": tensor[:,:,:,:,apexIdx]}

    return imgsDict


def getStatistics(data):
    mean       = np.mean(data)
    std        = np.std(data)
    mymin      = np.min(data)
    mymax      = np.max(data)
    median     = np.median(data)
    upQuart    = np.percentile(data, 75)
    lowQuart   = np.percentile(data, 25)
    iqr        = upQuart - lowQuart
    upWhisker  = data[data<=upQuart+1.5*iqr].max()
    lowWhisker = data[data>=lowQuart-1.5*iqr].min()
    return [mean, std, mymin, mymax, median, lowQuart, upQuart, lowWhisker, upWhisker]