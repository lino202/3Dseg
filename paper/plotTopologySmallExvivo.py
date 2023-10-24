import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import pandas as pd
import seaborn as sns


font = {'family' : "Times New Roman",
    'weight' : 'normal',
    'size'   : 20}
plt.rc('font', **font)
plt.rcParams.update({'mathtext.default':  'regular' })

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--samplePaths',  nargs='+',  type=str)
    parser.add_argument('--outPath',       type=str)
    parser.add_argument('--labels',       type=str, nargs='+')
    parser.add_argument('--ticks',       type=int)
    args = parser.parse_args()


    
    labels = args.labels
    betti_numbers = []
    classes = []
    values       = []

    for samplePath in args.samplePaths:


        with open(samplePath, 'rb') as f:
            results = pickle.load(f)
    
        
        phConstruction = samplePath.split('phconst')[-1].split('.')[0]
        if phConstruction == '0':
            labels = ["V-construction"]
        elif phConstruction == 'N':
            labels = ["T-construction"]

        for n,label in enumerate(labels):
            values = values + list(results[n,:,0])
            classes = classes + [label] * results.shape[1]
            betti_numbers = betti_numbers + ["$B_0$"] * results.shape[1]

            values = values + list(results[n,:,1])
            classes = classes + [label] * results.shape[1]
            betti_numbers = betti_numbers + ["$B_1$"] * results.shape[1]

            values = values + list(results[n,:,2])
            classes = classes + [label] * results.shape[1]
            betti_numbers = betti_numbers + ["$B_2$"] * results.shape[1]
    
    res = {"Values": values, "MYO": classes, "Betti Number": betti_numbers}
    resDF = pd.DataFrame(res)

    plt.figure(figsize=(5,3))
    sns.stripplot(
        data=resDF, x="MYO", y="Values", hue="Betti Number",
        dodge=True, alpha=.2, legend=False, jitter=0.35
    )
    err_kws = {"markersize":30, "markeredgewidth":4}
    sns.pointplot(
        data=resDF, x="MYO", y="Values", hue="Betti Number",
        dodge=.53, errorbar=("pi", 95), capsize=.15, linestyles="none", markers="o", err_kws=err_kws, legend=True
    )
    plt.grid()  
    ticks = list(np.linspace(1, np.nanmax(resDF.Values.to_numpy()), args.ticks).astype(int))
    plt.yticks(ticks)
    plt.ylabel('Betti Numbers')
    plt.legend()
    # plt.tight_layout()
    plt.savefig(os.path.join(args.outPath, "condense_topology.pdf"), dpi=300)


if __name__ == '__main__':
    main()
