''' Functions to get the ROI
    Code is based in Xiang Lin et al,  Automated Detection of Left Ventricle in 4D MR Images: 
    Experience from a Large Study
'''

import numpy as np
import scipy.fft
import scipy.ndimage as ndi
from scipy import optimize

def normalize(arr):
    if np.max(arr) - np.min(arr) != 0:
        arr = (arr - np.min(arr))/ (np.max(arr)-np.min(arr))
        return arr * 255
    else:
        return arr

def getH13D(img, medianFilt=True, threshold=0.05):
    slices = img.shape[3]
    res = np.zeros(img.shape[1:4])
    for s in range(slices):
        
        fft = scipy.fft.fftn(img[:,:,:,s])
        res[:,:,s] = np.absolute(scipy.fft.ifftn(fft[1, :, :])) 
 
    if medianFilt: res = ndi.median_filter(res, size = 5)
    res = normalize(res)
    res[res<threshold*np.max(res)] = 0
    return res


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    x,y = ndi.measurements.center_of_mass(data)
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def getROICINE(imgArr, plotPath=None):
    h1 = getH13D(imgArr)
    h1Sum = np.sum(h1, axis=2)
    # a = np.array([0,0])

    # while True:
    if plotPath:
        import matplotlib.pyplot as plt
        # plt.figure(),plt.subplot(131),plt.imshow(h1Sum, cmap='gray')
        # plt.subplot(132),plt.imshow(h1[:,:,6], cmap='gray')
        # plt.subplot(133),plt.imshow(imgArr[:,:,6,0], cmap='gray'), plt.show()
        plt.figure(),plt.imshow(h1Sum, cmap='gray', interpolation='none')

    params = fitgaussian(h1Sum)
    fit = gaussian(*params)
    gaussFit = fit(*np.indices(h1Sum.shape))
    
    if plotPath:
        plt.contour(gaussFit, cmap=plt.cm.copper)
        ax = plt.gca()
        plt.savefig(plotPath)
        plt.close()    
    
    (height, x, y, width_x, width_y) = params

    # if plotPath:
    #     plt.text(0.05, 0.03, """
    #     x : %.1f
    #     y : %.1f
    #     width_x : %.1f
    #     width_y : %.1f""" %(x, y, width_x, width_y),
    #             fontsize=16, horizontalalignment='left',
    #             verticalalignment='bottom', transform=ax.transAxes, color='w')
    #     plt.show()

    # b = np.array([x,y])
    # # print("Center is : {}".format(b))
    # if  np.linalg.norm(a-b) < 1:
    #     break
    # else:
    #     a = x, y
        
    # gaussFit = gaussFit / height
    # h1Sum[gaussFit<0.05] = 0

    return x, y, width_x*2, width_y*2 


def getROIMsk(arr):
    idxs = arr.nonzero()
    idxs = np.array(idxs)
    roi_min = np.min(idxs,1)
    roi_max = np.max(idxs,1)
    roi_max += 20
    roi_min -= 20
    idxstmp = np.array(arr.shape) - 1 < roi_max
    roi_max[idxstmp] = (np.array(arr.shape) - 1)[idxstmp]
    idxstmp = np.zeros(len(arr.shape)) > roi_min
    roi_min[idxstmp] = np.zeros(len(arr.shape))[idxstmp]
    return roi_min, roi_max