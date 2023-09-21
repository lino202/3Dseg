import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import persim




def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--samplePath',    type=str)
    parser.add_argument('--inputPath',    type=str)
    parser.add_argument('--outPath',       type=str)
    parser.add_argument('--slice',         type=int, nargs='+')
    args = parser.parse_args()


    font_size = 16
    font = {'family' : "Times New Roman",
        'weight' : 'normal',
        'size'   : font_size}
    plt.rc('font', **font)
    plt.rcParams.update({'mathtext.default':  'regular'})
    fontdict = {'fontsize': font_size}

    # Read input image logits before and after PH and barcodes
    img     = tio.ScalarImage(args.inputPath) #This is the image used as an input
    print(img.shape)
    
    with open(os.path.join(args.samplePath, "bcodes_0.pickle"), 'rb') as f:
        bcodes_first = pickle.load(f)
    with open(os.path.join(args.samplePath, "bcodes_99.pickle"), 'rb') as f:
        bcodes_last = pickle.load(f)
    with open(os.path.join(args.samplePath, "logits_0.pickle"), 'rb') as f:
        logits_first = pickle.load(f)
    with open(os.path.join(args.samplePath, "logits_99.pickle"), 'rb') as f:
        logits_last = pickle.load(f)
    with open(os.path.join(args.samplePath, "roi.pickle"), 'rb') as f:
        roi = pickle.load(f)
    
    
    img = img.data.numpy()[tuple(roi)]
    myslice = []
    for i, s in enumerate(args.slice):
        if s >= 0: myslice.append(s)
        if s < 0: myslice.append(slice(0,img.shape[i+1],1))

    for i_combo in range(len(bcodes_first)):

        combo_bcodes_first = bcodes_first[i_combo][((bcodes_first[i_combo][:,2] - bcodes_first[i_combo][:,1])>0.05).nonzero()[0], :]   # get lifetime over 0.05
        combo_bcodes_last = bcodes_last[i_combo][((bcodes_last[i_combo][:,2] - bcodes_last[i_combo][:,1])>0.05).nonzero()[0], :]
        myslice = [i_combo, *myslice]
        combo_logits_first = np.abs(logits_first[tuple(myslice)] - 1)
        combo_logits_last = np.abs(logits_last[tuple(myslice)] - 1)

        labels= ["$B_0$", "$B_1$", "$B_2$"] 

        fig, axs = plt.subplots(1, 2)
        pds_first = [combo_bcodes_first[combo_bcodes_first[:,0] == i] for i in range(3)]
        pds_first = [p[:,1:3]for p in pds_first]
        persim.plot_diagrams(pds_first, ax=axs[0], lifetime=False, xy_range=[0, 1.0, 0, 0.99], labels=labels, size=30)
        axs[0].set_title("Without PH postprocessing", **fontdict)
        axs[0].grid()

        pds_last = [combo_bcodes_last[combo_bcodes_last[:,0] == i] for i in range(1)]
        pds_last = [p[:,1:3]for p in pds_last]
        persim.plot_diagrams(pds_last, ax=axs[1], lifetime=False, xy_range=[0, 1.0, 0, 0.99], labels=labels, size=30)
        axs[1].set_title("With PH postprocessing", **fontdict)
        axs[1].set_yticklabels([])
        axs[1].set_ylabel('')
        axs[1].grid()

        fig.savefig(os.path.join(args.outPath, "PH_diagrams.pdf"), dpi=300)


        fig, axs = plt.subplots(1, 2, dpi=300)
        im = axs[0].imshow(combo_logits_first, vmin=0.1, vmax=0.9) 
        axs[0].imshow(img[tuple(myslice)], alpha=0.5, cmap='gray')
        im = axs[1].imshow(combo_logits_last, vmin=0.1, vmax=0.9)
        axs[1].imshow(img[tuple(myslice)], alpha=0.5, cmap='gray')
        axs[0].set_xticks([]), axs[0].set_yticks([])
        axs[1].set_xticks([]), axs[1].set_yticks([])
        cbar = plt.colorbar(im, ax=axs.ravel().tolist(), label='Probability')
        cbar.ax.set_ylabel("Prediction probability", **font)
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), **font)
        # plt.show()
        fig.savefig(os.path.join(args.outPath, "PH_logits.pdf"), dpi=300)

if __name__ == '__main__':
    main()
