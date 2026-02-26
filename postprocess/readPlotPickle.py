import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys 
sys.path.append(os.path.join('/'.join(sys.path[0].split("/")[:-1])))
from utils.util import getStatistics
from scipy.io import savemat
import pickle

sns.set(style='whitegrid')
font = {'family' : "Times New Roman",
    'weight' : 'bold',
    'size'   : 22}
plt.rc('font', **font)
plt.rcParams.update({'mathtext.default':  'regular' })

def groupFoldResults(paths):
    for i, foldPath in enumerate(paths):
        tmpData = pd.read_pickle(os.path.join(foldPath, "results.pickle"))
        if i==0: 
            classes = tmpData.keys()
            groupData = tmpData
            continue
        for c in classes:   
            groupData[c] = np.concatenate((groupData[c], tmpData[c]))

    return groupData

def map2DF(data, label):
    # This work as long as all keys have array associated with the same length
    label_change = {"lv": "LV", "myo": "MYO", "rv": "RV", 
                "lvmyo": "LV-MYO", "lvrv": "LV-RV", "lvmi": "LV-MI", 
                "myorv": "MYO-RV", "myomi": "MYO-MI",
                "mi": "MI", "be": "BE", "ts": "TS"} 
    tmpkey = list(data.keys())[0]
    n = data[tmpkey][list(data[tmpkey].keys())[0]].shape[0]
    mydict = {}
    mydict['Approach'] = []
    mydict[label]     = np.array([])
    mydict['Class']    = []
    for approach in data.keys():
        for key in data[approach].keys():
            if label.split(' ')[0].lower() in key and not 'bg' in key:
                # tmp = np.concatenate((dataBaseline[key], dataPH[key]))
                mydict[label] = np.concatenate((mydict[label], data[approach][key])) if mydict[label].size else data[approach][key]
                classe = label_change[key.split('_')[-1]]
                tmp = [classe] * n
                mydict['Class'] = [*mydict['Class'], *tmp]
                tmp = [approach] * n
                mydict['Approach'] = [*mydict['Approach'], *tmp]
    return mydict

def plot(data, label, resPath, fontsize=20):
    plt.figure(figsize=(8, 6), dpi=80)
    boxplot = sns.boxplot(data=data, x='Class', y=label, hue='Approach') #flierprops={'marker': 'd', 'markeredgecolor': 'none', 'markerfacecolor':'gray'}
    # boxplot.set_ylabel(label, fontsize=fontsize)
    boxplot.tick_params(labelsize=fontsize)
    if resPath:
        plt.savefig(os.path.join(resPath, "{}.pdf".format(label)))
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePath',  required=True, type=str)
    parser.add_argument('--nFold',     required=True, type=int)
    parser.add_argument('--resPath', type=str)
    args = parser.parse_args()
    
    #Collectand group data from all folds
    gan_baseline = [os.path.join(args.filePath, "fold_deform_{}".format(i), "metrics", "volumes_baseline_phconstN_originalSpace") for i in range(args.nFold)]
    gan_ph       = [os.path.join(args.filePath, "fold_deform_{}".format(i), "metrics", "volumes_ph_phconstN_originalSpace") for i in range(args.nFold)]
    baseline     = [os.path.join(args.filePath, "fold_{}".format(i), "metrics", "volumes_baseline_phconstN_originalSpace") for i in range(args.nFold)]
    ph           = [os.path.join(args.filePath, "fold_{}".format(i), "metrics", "volumes_ph_phconstN_originalSpace") for i in range(args.nFold)]
    
    data_gan_baseline = groupFoldResults(gan_baseline)
    data_gan_ph       = groupFoldResults(gan_ph)
    data_baseline     = groupFoldResults(baseline)
    data_ph           = groupFoldResults(ph)

    data = {"B": data_baseline, "TC": data_ph, "SA": data_gan_baseline, "SATC": data_gan_ph}
    
    # Save data to mat file and pickle file for later use
    savemat(os.path.join(args.resPath, "results.mat"), data) 
    with open(os.path.join(args.resPath, "results.pickle"), "wb") as f:
        pickle.dump(data, f)

    # #REMAP into DF--------------------------------------------
    # gdsc = map2DF(data, 'gDSC')
    # hd   = map2DF(data, 'HD (mm)')
    # assd = map2DF(data, 'ASSD (mm)')
    # be   = map2DF(data, 'BE')
    # ts   = map2DF(data, 'TS')
            
    # gdsc = pd.DataFrame(gdsc)
    # hd   = pd.DataFrame(hd)
    # assd = pd.DataFrame(assd)
    # be   = pd.DataFrame(be)
    # ts   = pd.DataFrame(ts)
    
    # #Save or plot ----------------------------------
    # plot(gdsc, 'gDSC', args.resPath)
    # plot(hd,   'HD (mm)', args.resPath)
    # plot(assd, 'ASSD (mm)', args.resPath)
    # plot(be,   'BE', args.resPath)
    # plot(ts,   'TS', args.resPath)
    
    # # Get overall statistics
    # res_dataframe = []
    # res_excel_indexs = []
    # nSamples = data_baseline['ts'].shape[0]
    # for approach in data.keys():
    #     for metric_class in data[approach].keys():
    #         res_excel_indexs.append("{}_{}".format(approach, metric_class)) 
    #         if 'ts' != metric_class:
    #                 res_dataframe.append(getStatistics(data[approach][metric_class]))        
    #         else:
    #                 tmp = np.ones(9+1) * np.nan
    #                 tmp[-1] = np.sum(data[approach][metric_class]) / nSamples
    #                 res_dataframe.append(tmp)

    # columns = ['mean', 'std', 'min', 'max', 'median', 'lowQuart', 'upQuart', 'lowWhisker', 'upWhisker', 'perc']
    # df = pd.DataFrame(res_dataframe, index=res_excel_indexs, columns=columns)

    # res_excel = os.path.join(args.resPath, "stats.xlsx")
    # if not os.path.exists(res_excel):
    #         df.to_excel(res_excel, sheet_name='sheet1')
    # else:   
    #         with pd.ExcelWriter(res_excel, engine="openpyxl", mode='a',if_sheet_exists="overlay") as writer:
    #                 startrow = writer.sheets['sheet1'].max_row
    #                 df.to_excel(writer, sheet_name='sheet1', startrow=startrow, header=False)
    

if __name__ == '__main__':
    main()