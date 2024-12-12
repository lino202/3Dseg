'''
We took the LGE dataset and erase all images and labels with do not have only lv, myo and mi
'''

import os
import argparse
import time
import torchio as tio

def getNotMI(dataPath):

    samples = sorted([x for x in os.listdir(dataPath)])
    
    sampleToErase = []
    for i, sample in enumerate(samples):

        print('preProcessing sample = {}'.format(sample))
        if "Emidec" in dataPath:
            imgPath = os.path.join(dataPath, sample, "Images", "{}.nii.gz".format(sample))
            mskPath = os.path.join(dataPath, sample, "Contours", "{}.nii.gz".format(sample))
        elif "Myosaiq" in dataPath or "LGE" in dataPath:
            newdataPath = "/".join(dataPath.split("/")[:-1])
            imgPath = os.path.join(newdataPath, "images", sample)
            mskPath = os.path.join(newdataPath, "labels", sample)
        else:
            raise ValueError("Dataset not implemented")
        subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))

        if subject.msk.data.numpy().max() !=3:
            sampleToErase.append(sample)
    return sampleToErase

def eraseFiles(dataPath, sample2Erase):

    for i, sample in enumerate(sample2Erase):

        print('Erasing sample = {}'.format(sample))
        if "Emidec" in dataPath:
            os.remove(os.path.join(dataPath, sample, "Images", "{}.nii.gz".format(sample)))
            os.remove(os.path.join(dataPath, sample, "Contours", "{}.nii.gz".format(sample)))
        elif "Myosaiq" in dataPath or "LGE" in dataPath:
            newdataPath = "/".join(dataPath.split("/")[:-1])
            os.remove(os.path.join(newdataPath, "images", sample))
            os.remove(os.path.join(newdataPath, "labels", sample))
        else:
            raise ValueError("Dataset not implemented")


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',   type=str)
    args = parser.parse_args()
    
    start = time.time()
    print("Processing samples at {}".format(args.filePath))
    sampleToErase = getNotMI(args.filePath)
    eraseFiles(args.filePath, sampleToErase)
    print("Total duration of processing: {} s ".format(time.time()-start))



if __name__ == '__main__':
    main()