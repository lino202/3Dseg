import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--samplePath',    type=str)
    parser.add_argument('--outPath',       type=str)
    parser.add_argument('--labels',       type=str, nargs='+')
    parser.add_argument('--ticks',       type=int)
    args = parser.parse_args()


    COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
              'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] #There is no more than ten priors never
    
    with open(args.samplePath, 'rb') as f:
        results = pickle.load(f)
    
    font = {'family' : "Times New Roman",
        'weight' : 'normal',
        'size'   : 10}
    plt.rc('font', **font)
    plt.rcParams.update({'mathtext.default':  'regular' })
    # plt.rcParams['figure.figsize'] = [15, 12]
    labels = args.labels
    x = np.arange(results.shape[1]) + 1
    for n,label in enumerate(labels):
        f, ax = plt.subplots(3,1, tight_layout=True)
        ax[0].scatter(x,results[n,:,0].T, c=COLORS[n], label=label)
        ax[1].scatter(x,results[n,:,1].T, c=COLORS[n], label=label)
        ax[2].scatter(x,results[n,:,2].T, c=COLORS[n], label=label)
        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        ax[2].legend(loc="upper right")
        ax[0].set_ylabel("$B_0$")
        ax[1].set_ylabel("$B_1$")
        ax[2].set_ylabel("$B_2$")

        ax[2].set_xlabel("Samples")
        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])

        ticks = list(x[::int(x.shape[0]/args.ticks)])
        ticks.append(x.max())
        ax[2].set_xticks(ticks)

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[0].set_xlim([x.min(), x.max()])
        ax[1].set_xlim([x.min(), x.max()])
        ax[2].set_xlim([x.min(), x.max()])
        fig = plt.gcf()
        fig.savefig(os.path.join(args.outPath, "{}.pdf".format(label)), dpi=400)
        plt.close()

if __name__ == '__main__':
    main()
