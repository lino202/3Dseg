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

	vols_preprocessed = args.rootPath
	if os.path.exists(args.resPath): shutil.rmtree(args.resPath)
	NFOLD = int(list(cvFolds['train'].keys())[-1].split("_")[-1])+1

	for i in range(NFOLD):
		trainPath = os.path.join(args.resPath, 'fold_{}'.format(i), 'train')
		if not os.path.exists(trainPath): pathlib.Path(trainPath).mkdir(parents=True, exist_ok=True)

		for item in cvFolds['train']['fold_{}'.format(i)]:
#			if "mi" in item:
			srcPath = os.path.join(vols_preprocessed, "{}_deform".format(item), "msk.nii")
			dstPath = os.path.join(trainPath, "{}_deform".format(item))
			if not os.path.exists(dstPath): pathlib.Path(dstPath).mkdir(parents=True, exist_ok=True)
			dstPath = os.path.join(dstPath, "msk.nii")
			#we make a symlink of only the msk for no overwritting the pred image from GANs trained from different folds
			os.symlink(srcPath, dstPath)
			print("Making link src: {}, dst {}".format(srcPath, dstPath))

if __name__ == '__main__':
	main()
