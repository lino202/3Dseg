'''Here we get the symlinks to the raw data from the 'all' folder and put them into the nnUNet format. 
We also create the GAN-augmented datasets with the original space and resolutions. This was tested in the LGE dataset

In this case we copy the data instead of creating symlinks because we launch the nnUNet in linux and the data creation is done in windows, so we try to avoid issues in this regard,
feel free to change this if you are working in linux and want to create symlinks instead.

We use the CV folders and sweep thorugh all folds and get the data, we do this because we decided to use 
the processed volumens as the deform GAN-agumented data is not in the orginal space-resolution and we will need to reconvert it
to add it to the nnUNet. For this reason we collect the data from the CV folders, the nnUNet will trained and from its validation dataset we will get the prediction and then 
we will reconvert the validation dataset to the original space and resolution and we will get the metrics for the predictions '''

import shutil
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inFolder', type=str, required=True)
    parser.add_argument('--outFolder', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--folds', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()

    out_folder = os.path.join(args.outFolder, "Dataset00{}_Myosaiq{}".format(1 if args.dataset!='SA' else 2, 'GAN' if args.dataset=='SA' else ''))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    if not os.path.exists(os.path.join(out_folder, 'imagesTr')):
        os.makedirs(os.path.join(out_folder, 'imagesTr'), exist_ok=True)
    if not os.path.exists(os.path.join(out_folder, 'labelsTr')):
        os.makedirs(os.path.join(out_folder, 'labelsTr'), exist_ok=True)
    # if (args.mode == 'test') and (not os.path.exists(os.path.join(out_folder, 'imagesTs'))):
    #     os.makedirs(os.path.join(out_folder, 'imagesTs'), exist_ok=True)

    for fold in range(args.folds):
        foldPath = os.path.join(args.inFolder, "fold_{}".format(fold))
        
        # First we get the data on train
        files_train = os.listdir(os.path.join(foldPath,'train'))
        files_train = [os.path.join(foldPath, 'train', file) for file in files_train]
        files_val = os.listdir(os.path.join(foldPath,'val'))
        files_val = [os.path.join(foldPath, 'val', file) for file in files_val]
        files = files_train + files_val
        
        for file in files:
            src_img = os.path.join(file, 'img.nii')
            if args.mode == 'train':
                src_msk = os.path.join(file, 'msk.nii')
            fileName = os.path.basename(file)

            if args.mode == 'train':
                dst_img = os.path.join(out_folder, 'imagesTr', "{}_0000.nii".format(fileName))
                dst_msk = os.path.join(out_folder, 'labelsTr', "{}.nii".format(fileName))
            else:
                dst_img = os.path.join(out_folder, 'imagesTs', "{}_0000.nii".format(fileName))

            print(src_img, dst_img)
            shutil.copy(src_img, dst_img)
            if args.mode == 'train':
                print(src_msk, dst_msk)
                shutil.copy(src_msk, dst_msk)
        
        