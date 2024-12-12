'''This compute BE for assessing different approaches
between msk and preds, which could be or not on the original space
If in the original space, you will use the msks before preprocessing 
so be carefull to replicate the msks preprocessing that affected labels (not shaping)'''    

import os
import argparse
import time
import torch.nn.functional as F
import numpy as np
import torchio as tio
import sys
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
        from utils.topo import BEmetricSeparated
        from utils.util import getStatistics

        #Start
        samples = sorted([x for x in os.listdir(args.predsFolder)])
        nSamples = len(samples)
        print('There are {} samples in total'.format(len(samples)))

        # Get prior, this important when using and not using PH as the BE metric needs this to compute the difference in 
        # Topo between the msks and preds
        if   "CINE"   in args.priorName: prior = PRIOR_CINE
        elif "EXVIVO" in args.priorName: prior = PRIOR_EXVIVO
        elif "LGE"    in args.priorName: prior = PRIOR_LGE
        else: raise ValueError("Wrong priorName")
        
        if args.phThres < 0: phThres = None
        else: phThres = args.phThres

        be_0   = np.zeros((nSamples, len(prior)))
        be_1   = np.zeros((nSamples, len(prior)))
        be_2   = np.zeros((nSamples, len(prior)))
        for i, sample in enumerate(samples):

                print('Processing sample = {}'.format(sample))
                start_iter = time.time()
                
                #Get img, msk and pred in a subject and check consistency
                img, msk  = getSubjectImgMsk(getEXFromMask, args.imgsFolder, args.msksFolder, sample)
                predPath = os.path.join(args.predsFolder, sample, "pred.nii".format(sample))
                pred     = tio.LabelMap(predPath)

                subject = tio.Subject(img=img, msk=msk, pred=pred)
                subject.check_consistent_attribute('spacing') 
                # subject.check_consistent_attribute('affine') # this might give error due to rounding error
                subject.check_consistent_attribute('shape')

                #Get msk and pred one-hot for compute metrics
                one_hot = F.one_hot(pred.data.long(), num_classes=args.nClasses)
                pred_one_hot = one_hot.permute(0, 4, 1, 2, 3).type(pred.data.type())
                
                one_hot = F.one_hot(msk.data.long(), num_classes=args.nClasses)
                msk_one_hot = one_hot.permute(0, 4, 1, 2, 3).type(msk.data.type())
                
                #Get BE separated
                be_0[i,:], be_1[i,:], be_2[i,:] = BEmetricSeparated(pred_one_hot[0,:,:,:,:], msk_one_hot[0,:,:,:,:], prior, args.phParallel)
                
                #Print info
                print("Processed sample {}/{} took {} s".format(i+1, nSamples, time.time() - start_iter))


        #Save general results ---------------------------------------------------------------------------
        #Save per volumes parameters results  
        print("Saving results -------------------------------")

        be = np.concatenate((be_0[:,np.newaxis], be_1[:,np.newaxis], be_2[:,np.newaxis]), axis=1)
        res_dict = {}
        for i, index_name in enumerate(args.res_excel_indexs):
                for j in range(3):
                        res_dict["{}_{}".format(index_name,j)] = be[:,j,i]
        with open(os.path.join(args.resPath, "results_separatedBE.pickle"), 'wb') as f:
                pickle.dump(res_dict, f)
    
        #Save statiscal summary on excel with exp name
        exp_name = args.predsFolder.split('/')[-1]
        res_excel = os.path.join(args.resPath, "stats_separateBE.xlsx")
    
        res_dataframe = []
        for i, index_name in enumerate(args.res_excel_indexs):
                for j in range(3):
                        res_dataframe.append(getStatistics(be[:,j,i]))        
    
        indexs = []
        for index in args.res_excel_indexs:
                for j in range(3):
                        indexs.append("{}_{}_{}".format(exp_name, index, j)) 
        columns = ['mean', 'std', 'min', 'max', 'median', 'lowQuart', 'upQuart', 'lowWhisker', 'upWhisker']
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