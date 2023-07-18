'''
Read cross validation distribution and get symlinks and folder structure
if in windows activate developer mode for win10+ versions
'''


import os
import argparse
import pickle
import pathlib
import shutil

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--cvPickleFilePath',   type=str)
    parser.add_argument('--rootPath',   type=str)
    parser.add_argument('--resPath',    type=str)
    args = parser.parse_args()
    
    with open(args.cvPickleFilePath, 'rb') as handle:
        cvFolds = pickle.load(handle)

    rootPath = args.rootPath
    vols_preprocessed = os.path.join(rootPath, 'vols_preprocessed')
    if os.path.exists(args.resPath): shutil.rmtree(args.resPath)
    NFOLD = int(list(cvFolds['train'].keys())[-1].split("_")[-1])+1

    for i in range(NFOLD):
        trainPath = os.path.join(args.resPath, 'fold_{}'.format(i), 'train')
        valPath = os.path.join(args.resPath, 'fold_{}'.format(i), 'val')
        if not os.path.exists(trainPath): pathlib.Path(trainPath).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(valPath): pathlib.Path(valPath).mkdir(parents=True, exist_ok=True)

        for item in cvFolds['train']['fold_{}'.format(i)]:
            # if "mi" in item:
            srcPath = os.path.join(vols_preprocessed, item)
            dstPath = os.path.join(trainPath, item)
            os.symlink(srcPath, dstPath, target_is_directory=True)
            print("Making link src: {}, dst {}".format(srcPath, dstPath))

        for item in cvFolds['val']['fold_{}'.format(i)]:
            # if "mi" in item:
            srcPath = os.path.join(vols_preprocessed, item)
            dstPath = os.path.join(valPath, item)
            os.symlink(srcPath, dstPath, target_is_directory=True)
            print("Making link src: {}, dst {}".format(srcPath, dstPath))

if __name__ == '__main__':
    main()