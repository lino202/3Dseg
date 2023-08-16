import sys 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
sys.path.append(os.path.join('/'.join(sys.path[0].split("/")[:-1])))
from utils.util import getStatistics

sns.set(style='whitegrid')


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
    tmpkey = list(data.keys())[0]
    n = data[tmpkey][list(data[tmpkey].keys())[0]].shape[0]
    mydict = {}
    mydict['approach'] = []
    mydict[label]     = np.array([])
    mydict['class']    = []
    for approach in data.keys():
        for key in data[approach].keys():
            if label.split(' ')[0].lower() in key and not 'bg' in key:
                # tmp = np.concatenate((dataBaseline[key], dataPH[key]))
                mydict[label] = np.concatenate((mydict[label], data[approach][key])) if mydict[label].size else data[approach][key]
                tmp = [key.split('_')[-1]] * n
                mydict['class'] = [*mydict['class'], *tmp]
                tmp = [approach] * n
                mydict['approach'] = [*mydict['approach'], *tmp]
    return mydict

def plot(data, label, resPath):
    plt.figure()
    sns.boxplot(data=data, x='class', y=label, hue='approach')
    if resPath:
        plt.savefig(os.path.join(resPath, "{}.png".format(label)))
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
    gan_baseline = [os.path.join(args.filePath, "base_deform_fold_{}".format(i), "metrics", "volumes_baseline_phconstN_originalSpace") for i in range(args.nFold)]
    gan_ph       = [os.path.join(args.filePath, "base_deform_fold_{}".format(i), "metrics", "volumes_ph_phconstN_originalSpace") for i in range(args.nFold)]
    baseline     = [os.path.join(args.filePath, "fold_{}".format(i), "metrics", "volumes_baseline_phconstN_originalSpace") for i in range(args.nFold)]
    ph           = [os.path.join(args.filePath, "fold_{}".format(i), "metrics", "volumes_ph_phconstN_originalSpace") for i in range(args.nFold)]
    
    data_gan_baseline = groupFoldResults(gan_baseline)
    data_gan_ph       = groupFoldResults(gan_ph)
    data_baseline     = groupFoldResults(baseline)
    data_ph           = groupFoldResults(ph)

    data = {"baseline": data_baseline, "ph": data_ph, "gan_baseline": data_gan_baseline, "gan_ph": data_gan_ph}
    
    #REMAP into DF--------------------------------------------
    gdsc = map2DF(data, 'gDSC')
    hd   = map2DF(data, 'HD (mm)')
    assd = map2DF(data, 'ASSD (mm)')
    be   = map2DF(data, 'BE')
    ts   = map2DF(data, 'TS')
            
    gdsc = pd.DataFrame(gdsc)
    hd   = pd.DataFrame(hd)
    assd = pd.DataFrame(assd)
    be   = pd.DataFrame(be)
    ts   = pd.DataFrame(ts)
    
    #Save or plot ----------------------------------
    plot(gdsc, 'gDSC', args.resPath)
    plot(hd,   'HD (mm)', args.resPath)
    plot(assd, 'ASSD (mm)', args.resPath)
    plot(be,   'BE', args.resPath)
    plot(ts,   'TS', args.resPath)
    
    # Get overall statistics
    res_dataframe = []
    res_excel_indexs = []
    nSamples = data_baseline['ts'].shape[0]
    for approach in data.keys():
        for metric_class in data[approach].keys():
            res_excel_indexs.append("{}_{}".format(approach, metric_class)) 
            if 'ts' != metric_class:
                    res_dataframe.append(getStatistics(data[approach][metric_class]))        
            else:
                    tmp = np.ones(9+1) * np.nan
                    tmp[-1] = np.sum(data[approach][metric_class]) / nSamples
                    res_dataframe.append(tmp)

    columns = ['mean', 'std', 'min', 'max', 'median', 'lowQuart', 'upQuart', 'lowWhisker', 'upWhisker', 'perc']
    df = pd.DataFrame(res_dataframe, index=res_excel_indexs, columns=columns)

    res_excel = os.path.join(args.resPath, "stats.xlsx")
    if not os.path.exists(res_excel):
            df.to_excel(res_excel, sheet_name='sheet1')
    else:   
            with pd.ExcelWriter(res_excel, engine="openpyxl", mode='a',if_sheet_exists="overlay") as writer:
                    startrow = writer.sheets['sheet1'].max_row
                    df.to_excel(writer, sheet_name='sheet1', startrow=startrow, header=False)
    

if __name__ == '__main__':
    main()