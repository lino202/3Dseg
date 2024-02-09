"""Here we transform the volume to have the original shape and spacing
respecting the initial spacial orientation. Here we only work the original msk
and the prediction. 
"""

import argparse
import pickle
import time
import torchio as tio


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--predPath',   type=str, required=True)
    parser.add_argument('--imgOriPath', type=str, required=True)
    parser.add_argument('--subjectPath', type=str, required=True)
    parser.add_argument('--resPath',      type=str)
    args = parser.parse_args()

    # Get pred and subject when this was preprocessed
    # we get the subjects-preprocessed to invert crop and padding on prediction 
    pred = tio.LabelMap(args.predPath)
    img  = tio.ScalarImage(args.imgOriPath)
    
    with open(args.subjectPath, 'rb') as f: preprocessedSubject = pickle.load(f)

    print("Initial shape and spacing")
    print("Pred {}".format(pred))

    #Transform
    #Padding and crop
    inverse_transform = preprocessedSubject.get_inverse_transform(warn=False)
    pred = inverse_transform(pred)

    #Final resampling, this adds uncertainty
    trans = tio.Resample(img)
    pred = trans(pred)

    subject = tio.Subject(img=img, pred=pred)
    trans   = tio.CopyAffine('img')
    subject = trans(subject)
    subject.check_consistent_attribute('spacing') 
    subject.check_consistent_attribute('affine')
    subject.check_consistent_attribute('shape')

    print("Final shape and spacing")
    print("Img {}\nPred {}".format(img, pred))

    #Save
    subject.pred.save(args.resPath, squeeze=True)

if __name__ == '__main__':
    start = time.time()
    main()
    print("Total duration of processing: {} s ".format(time.time()-start))

