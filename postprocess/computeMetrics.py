'''This compute different metrics for assessing different approaches
between msk and preds, which could be or not on the original space
If in the original space, you will use the msks before preprocessing 
so be carefull to replicate the msks preprocessing that affected labels (not shaping)'''    

import os
import argparse
import pathlib
import time
import torch.nn.functional as F
import numpy as np
import torchio as tio
import sys
import monai
import torch
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def getSubjectImgMsk(getEXFromMask, imgsFolder, msksFolder, sample):
        """This function search for the the img and msk of the subject in the original space or in the preprocessed one
        Also, if we use the original msks we need to replicate the preprocessing applied to the labels, as some examples might have changed.
        For example, one example in the MnM dataset had two islands for the RV which was cleaned in preprocessing, the Mvo is delete it in LGE, and so on.. """

        if "vols_preprocessed" in imgsFolder: 
                # If we used the preprocessed the reading is so much easier but we're not using the original space
                imgPath = os.path.join(imgsFolder, sample, "img.nii")
                mskPath = os.path.join(msksFolder, sample, "msk.nii")
                subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
        else:
                if "MnM" in imgsFolder:
                        imgPath = os.path.join(imgsFolder, sample, "{}_sa.nii.gz".format(sample))
                        mskPath = os.path.join(msksFolder, sample, "{}_sa_gt.nii.gz".format(sample))
                        subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
                        edes    = getEXFromMask(subject.msk.data.numpy(), sample)

                        #Keep biggest island
                        trans = tio.KeepLargestComponent()
                        tmpED = subject.msk.data[edes[0],:,:,:] # we are only interest in ED
                        tmpED = F.one_hot(tmpED.long(), num_classes=int(subject.msk.data.max())+1)
                        tmpED = tmpED.permute(3, 0, 1, 2).type(tmpED.type())
                        for fg in range(1,tmpED.shape[0]):
                                subjectMsk = tio.Subject(msk=tio.LabelMap(tensor=tmpED[fg][np.newaxis,:]))
                                subjectMsk = trans(subjectMsk)
                                tmpED[fg] = subjectMsk.msk.data
                        tmpED = tmpED.argmax(dim=0)
                        subject.msk.data[edes[0]] = tmpED

                        subject.msk.set_data(subject.msk.data[edes[0]][np.newaxis, :])
                        subject.img.set_data(subject.img.data[edes[0]][np.newaxis, :])
                
                elif "Myosaiq" in imgsFolder or "LGE" in imgsFolder:
                        origSampleName = "_".join(sample.split('_')[:-1])
                        imgPath = os.path.join(imgsFolder, "{}.nii.gz".format(origSampleName))
                        mskPath = os.path.join(msksFolder, "{}.nii.gz".format(origSampleName))
                        subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))
                        subject.msk.data[subject.msk.data>3] = 3  #This is only done for this datasets

                elif "Exvivo" in imgsFolder:
                        imgPath = os.path.join(imgsFolder, sample, "img.nii")
                        mskPath = os.path.join(msksFolder, sample, "msk.nii")
                        subject = tio.Subject(img=tio.ScalarImage(imgPath), msk=tio.LabelMap(mskPath))

                        #Keep biggest island
                        trans = tio.KeepLargestComponent()
                        tmp = subject.msk.data[0] 
                        tmp = F.one_hot(tmp.long(), num_classes=int(subject.msk.data.max())+1)
                        tmp = tmp.permute(3, 0, 1, 2).type(tmp.type())
                        for fg in range(1,tmp.shape[0]):
                                subjectMsk = tio.Subject(msk=tio.LabelMap(tensor=tmp[fg][np.newaxis,:]))
                                subjectMsk = trans(subjectMsk)
                                tmp[fg] = subjectMsk.msk.data
                        tmp = tmp.argmax(dim=0)
                        subject.msk.data[0] = tmp

                elif "Emidec" in imgsFolder:
                        raise ValueError("The Emidec Dataset is not use directly")
                else:
                        raise ValueError("Dataset not implemented")
        return subject.img, subject.msk


def main():
        parser = argparse.ArgumentParser(description="Options")
        parser.add_argument('--predsFolder',  type=str, required=True)
        parser.add_argument('--msksFolder',   type=str, required=True)
        parser.add_argument('--imgsFolder',   type=str, required=True)
        parser.add_argument('--nClasses',     type=int,                help='number of classes in the output, including the background')
        parser.add_argument('--priorName',    type=str,                help='prior to use, specific to the dataset')
        parser.add_argument('--phThres',      type=float, default=0.5, help='threshold in the image, usually this is 0.5 as we hot-one encode the pred')
        parser.add_argument('--phParallel',   action='store_true',  help='use parallel calculation of PH')
        parser.add_argument('--resPath',      type=str)
        parser.add_argument('--res_excel_indexs', type=str, nargs='+', help='indexes for saving params results in the excel file and saving labels to the results CAREFUL!')
        parser.add_argument('--rootCodePath', type=str)
        args = parser.parse_args()

        sys.path.append(args.rootCodePath)
        from preprocess.utilsPre import getEXFromMask
        from utils.priors import PRIOR_CINE, PRIOR_EXVIVO, PRIOR_LGE
        from utils.topo import BEmetric
        from utils.util import getStatistics

        #Start
        samples = sorted([x for x in os.listdir(args.predsFolder)])
        nSamples = len(samples)
        print('There are {} samples in total'.format(len(samples)))

        plots_path     = os.path.join(args.resPath, "plots_vols")
        if not os.path.exists(plots_path): pathlib.Path(plots_path).mkdir(parents=True, exist_ok=True)

        # Get prior, this important when using and not using PH as the BE metric needs this to compute the difference in 
        # Topo between the msks and preds
        if   "CINE"   in args.priorName: prior = PRIOR_CINE
        elif "EXVIVO" in args.priorName: prior = PRIOR_EXVIVO
        elif "LGE"    in args.priorName: prior = PRIOR_LGE
        else: raise ValueError("Wrong priorName")
        
        if args.phThres < 0: phThres = None
        else: phThres = args.phThres

        gdsc = np.zeros((nSamples, args.nClasses))
        hd   = np.zeros((nSamples, args.nClasses))
        assd = np.zeros((nSamples, args.nClasses))
        ts   = np.zeros(nSamples)
        be   = np.zeros(nSamples)
        for i, sample in enumerate(samples):

                print('Processing sample = {}'.format(sample))
                start_iter = time.time()
                
                #Get img, msk and pred in a subject and check consistency
                img, msk  = getSubjectImgMsk(getEXFromMask, args.imgsFolder, args.msksFolder, sample)
                predPath = os.path.join(args.predsFolder, sample, "pred.nii".format(sample))
                pred     = tio.LabelMap(predPath)

                subject = tio.Subject(img=img, msk=msk, pred=pred)
                subject.check_consistent_attribute('spacing') 
                # subject.check_consistent_attribute('affine') # this might be give error due to rounding error
                subject.check_consistent_attribute('shape')

                #Get msk and pred one-hot for compute metrics
                one_hot = F.one_hot(pred.data.long(), num_classes=args.nClasses)
                pred_one_hot = one_hot.permute(0, 4, 1, 2, 3).type(pred.data.type())
                
                one_hot = F.one_hot(msk.data.long(), num_classes=args.nClasses)
                msk_one_hot = one_hot.permute(0, 4, 1, 2, 3).type(msk.data.type())
                
                #Get gDSC, HD, BE and TS
                #there is an error on gDSC implementation as results has not shape [BxC]
                #For this reason we permute CxB and get background value in order to have the right values
                #Also it does not work when tensor are on cuda, they already submitted a PR.
                #TODO This should be checked or an issue should be raisen in https://github.com/Project-MONAI/MONAI
                gdsc[i,:] = monai.metrics.compute_generalized_dice(torch.permute(pred_one_hot, (1,0,2,3,4)), torch.permute(msk_one_hot, (1,0,2,3,4)), include_background=True)
                hd[i,:]   = monai.metrics.compute_hausdorff_distance(pred_one_hot, msk_one_hot, include_background=True, spacing=pred.spacing) #monai 1.2.0 admits spacing
                assd[i,:] = monai.metrics.compute_average_surface_distance(pred_one_hot, msk_one_hot, symmetric=True, include_background=True, spacing=pred.spacing)
                be[i]     = BEmetric(pred_one_hot[0,:,:,:,:], msk_one_hot[0,:,:,:,:], prior, args.phParallel)
                if be[i] == 0.: ts[i] = 1

                #Get arrays for plottings
                mskArr  = msk.data[0].numpy()
                predArr = pred.data[0].numpy()
                imgArr  = img.data[0].numpy()
                
                #Save per volume results
                #Save images, we disregard the first and last image in the stack as usually are completly background
                nImgs = 5
                f, ax = plt.subplots(3,nImgs)
                sliceIdxs = np.linspace(0,img.shape[-1]-1,nImgs+2)
                sliceIdxs = np.round(sliceIdxs[1:-1]).astype(int)
                for j, s in enumerate(sliceIdxs):       
                    ax[0,j].imshow(imgArr[:,:,s],  vmin=0, vmax=imgArr.max(), interpolation='none')
                    ax[0,j].axis('off')
                    ax[1,j].imshow(mskArr[:,:,s],  vmin=0, vmax=args.nClasses-1, interpolation='none')
                    ax[1,j].axis('off')
                    ax[2,j].imshow(predArr[:,:,s], vmin=0, vmax=args.nClasses-1, interpolation='none')
                    ax[2,j].axis('off')
                plt.savefig(os.path.join(plots_path, "{}.png".format(sample)), dpi=300)
                plt.close()
                
                #Print info
                print("Processed sample {}/{} took {} s".format(i+1, nSamples, time.time() - start_iter))


        #Save general results ---------------------------------------------------------------------------

        #Save per volumes parameters results  
        print("Saving results -------------------------------")

        res_params = np.vstack((gdsc.T, hd.T, assd.T, be, ts))
        res_dict = {}
        for i, index_name in enumerate(args.res_excel_indexs):
                res_dict[index_name] = res_params[i,:]
        with open(os.path.join(args.resPath, "results.pickle"), 'wb') as f:
                pickle.dump(res_dict, f)
    
        #Save statiscal summary on excel with exp name
        exp_name = args.predsFolder.split('/')[-1]
        res_excel = os.path.join(args.resPath, "stats.xlsx")
    
        res_dataframe = []
        for i, index_name in enumerate(args.res_excel_indexs):
                if 'ts' != index_name:
                        res_dataframe.append(getStatistics(res_params[i,:]))        
                else:
                        tmp = np.ones(9+1) * np.nan
                        tmp[-1] = np.sum(ts) / nSamples
                        res_dataframe.append(tmp)
    
        indexs  = ["{}_{}".format(exp_name, index) for index in args.res_excel_indexs] 
        columns = ['mean', 'std', 'min', 'max', 'median', 'lowQuart', 'upQuart', 'lowWhisker', 'upWhisker', 'perc']
        df = pd.DataFrame(res_dataframe, index=indexs, columns=columns)
    
        if not os.path.exists(res_excel):
                df.to_excel(res_excel, sheet_name='sheet1')
        else:   
                with pd.ExcelWriter(res_excel, engine="openpyxl", mode='a',if_sheet_exists="overlay") as writer:
                        startrow = writer.sheets['sheet1'].max_row
                        df.to_excel(writer, sheet_name='sheet1', startrow=startrow, header=False)


if __name__ == '__main__':
        start = time.time()
        main()
        print("Total duration of processing: {} s ".format(time.time()-start))