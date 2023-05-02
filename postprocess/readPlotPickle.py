import pickle 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')

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
            if label.lower() in key and not 'bg' in key:
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
    parser.add_argument('--filePaths',  nargs='+', required=True, type=str)
    parser.add_argument('--classes',    nargs='+', required=True, type=str)
    parser.add_argument('--resPath', type=str)
    args = parser.parse_args()
    
    data = {}
    for i, key in enumerate(args.classes):
        data[key] = pd.read_pickle(args.filePaths[i])
    
    # print((dataPH["hd_lv"]==np.inf).nonzero())
    # print((dataPH["hd_myo"]==np.inf).nonzero())
    # print((dataPH["hd_rv"]==np.inf).nonzero())
    
    #REMAP into DF--------------------------------------------
    gdsc = map2DF(data, 'gDSC')
    hd   = map2DF(data, 'HD')
    be   = map2DF(data, 'BE')
    ts   = map2DF(data, 'TS')
            
    gdsc = pd.DataFrame(gdsc)
    hd   = pd.DataFrame(hd)
    be   = pd.DataFrame(be)
    ts   = pd.DataFrame(ts)
    
    #Save or plot ----------------------------------
    plot(gdsc, 'gDSC', args.resPath)
    plot(hd, 'HD', args.resPath)
    plot(be, 'BE', args.resPath)
    plot(ts, 'TS', args.resPath)
    
    
    

if __name__ == '__main__':
    main()