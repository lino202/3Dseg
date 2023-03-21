import pickle 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')

def map2DF(dataBaseline, dataPH, label):
    n = dataBaseline[list(dataBaseline.keys())[0]].shape[0]
    mydict = {}
    mydict['approach'] = []
    mydict[label]     = np.array([])
    mydict['class']    = []
    for key in dataBaseline.keys():
        if label.lower() in key and not 'bg' in key:
            tmp = np.concatenate((dataBaseline[key], dataPH[key]))
            mydict[label] = np.concatenate((mydict[label], tmp)) if mydict[label].size else tmp
            tmp = [key.split('_')[-1]] * n * 2
            mydict['class'] = [*mydict['class'], *tmp]
            tmp = [*['baseline'] * n, *['ph'] * n]
            mydict['approach'] = [*mydict['approach'], *tmp]
    return mydict

def plot(data, label, resPath='0'):
    plt.figure()
    sns.boxplot(data=data, x='class', y=label, hue='approach')
    if resPath != '0':
        plt.savefig(os.path.join(resPath, "{}.png".format(label)))
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--filePathBaseline', required=True, type=str)
    parser.add_argument('--filePathPH',       required=True, type=str)
    parser.add_argument('--resPath',          required=True, type=str)
    args = parser.parse_args()
    
    dataBaseline = pd.read_pickle(args.filePathBaseline)
    dataPH = pd.read_pickle(args.filePathPH)
    # print((dataPH["hd_lv"]==np.inf).nonzero())
    # print((dataPH["hd_myo"]==np.inf).nonzero())
    # print((dataPH["hd_rv"]==np.inf).nonzero())
    
    #REMAP into DF--------------------------------------------
    gdsc = map2DF(dataBaseline, dataPH, 'gDSC')
    hd = map2DF(dataBaseline, dataPH, 'HD')
    be = map2DF(dataBaseline, dataPH, 'BE')
    ts = map2DF(dataBaseline, dataPH, 'TS')
            
    gdsc = pd.DataFrame(gdsc)
    hd = pd.DataFrame(hd)
    be = pd.DataFrame(be)
    ts = pd.DataFrame(ts)
    
    #Save or plot ----------------------------------
    plot(gdsc, 'gDSC', args.resPath)
    plot(hd, 'HD', args.resPath)
    plot(be, 'BE', args.resPath)
    plot(ts, 'TS', args.resPath)
    
    
    

if __name__ == '__main__':
    main()