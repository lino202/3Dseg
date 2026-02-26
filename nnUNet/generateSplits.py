'''We generate the splits for using exactly the same splits and 5-fold Cross Validation as for our baseline UNet case'''


import os
import shutil
from pathlib import Path
from typing import List

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
import numpy as np



# def create_ACDC_split(labelsTr_folder: str, seed: int = 1234) -> List[dict[str, List]]:
#     # labelsTr_folder = '/home/isensee/drives/gpu_data_root/OE0441/isensee/nnUNet_raw/nnUNet_raw_remake/Dataset027_ACDC/labelsTr'
#     nii_files = nifti_files(labelsTr_folder, join=False)
#     patients = np.unique([i[:len('patient000')] for i in nii_files])
#     rs = np.random.RandomState(seed)
#     rs.shuffle(patients)
#     splits = []
#     for fold in range(5):
#         val_patients = patients[fold::5]
#         train_patients = [i for i in patients if i not in val_patients]
#         val_cases = [i[:-7] for i in nii_files for j in val_patients if i.startswith(j)]
#         train_cases = [i[:-7] for i in nii_files for j in train_patients if i.startswith(j)]
#         splits.append({'train': train_cases, 'val': val_cases})
#     return splits



if __name__ == "__main__":
    
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument("--input_folder", type=str, required=True, help="The file were the splits were saved")
     parser.add_argument("--dataset_id", required=False, type=int)
     parser.add_argument("--folds", required=False, type=int, default=5)
     args = parser.parse_args()


     dataset_name = f"Dataset{args.dataset_id:03d}_{'B' if args.dataset_id != 2 else 'SA'}"

     splits = []
     for fold in range(args.folds):
          print(f"Generating split for fold {fold}...")  

          fold_path = os.path.join(args.input_folder, f"fold_{fold}")
          train_cases = os.listdir(os.path.join(fold_path, "train"))
          val_cases = os.listdir(os.path.join(fold_path, "val"))
          
          splits.append({'train': train_cases, 'val': val_cases})

     preprocessed_folder = os.path.join(nnUNet_preprocessed, dataset_name)
     save_json(splits, os.path.join(preprocessed_folder, 'splits_final.json'), sort_keys=False)

