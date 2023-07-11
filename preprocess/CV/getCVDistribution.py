'''
Get datasets cross validation file
'''

import os
import argparse
import numpy as np
from sklearn.model_selection import KFold
import pickle

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',   type=str, required=True)
    parser.add_argument('--resPath',    type=str, required=True)
    parser.add_argument('--dataTypes',  type=str, nargs='+', required=True)
    parser.add_argument('--NFOLD',      type=int, required=True)
    args = parser.parse_args()

    NFOLD = args.NFOLD
    X = list(np.arange(NFOLD))
    allSamples = sorted([x for x in os.listdir(args.filePath)])
    samples = {}
    
    if len(args.dataTypes) > 1:
        for dataType in args.dataTypes:
            tmp = np.array(list(filter(lambda s: dataType in s, allSamples)))
            samples[dataType] = np.array_split(tmp,NFOLD)
    else:
        samples[args.dataTypes[0]] = np.array_split(np.array(allSamples),NFOLD)

    cvFolds = {'train':{}, 'val':{}}
    for fold in range(NFOLD):
        cvFolds['train']["fold_{}".format(fold)] = []
        cvFolds['val']["fold_{}".format(fold)]   = []

    kf = KFold(n_splits=NFOLD)
    for fold, (train, val) in enumerate(kf.split(X)):
        for dataType in args.dataTypes:
            tmpTrain = [samples[dataType][x] for x in train]
            for sublist in tmpTrain:
                for item in sublist:
                    cvFolds['train']["fold_{}".format(fold)].append(item)
            
            tmpVal   = samples[dataType][val[0]]
            tmpVal   = [item for item in tmpVal]
            for item in tmpVal:  #only list not list of lists
                cvFolds['val']["fold_{}".format(fold)].append(item)
    
    with open(os.path.join(args.resPath, "cv_{}Fold.pickle".format(NFOLD)), 'wb') as f:
        pickle.dump(cvFolds, f)

if __name__ == '__main__':
    main()