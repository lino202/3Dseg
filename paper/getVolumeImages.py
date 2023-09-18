import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--samplePath',    type=str)
    parser.add_argument('--outPath',       type=str)
    parser.add_argument('--slice',         type=int, nargs='+')
    args = parser.parse_args()

    img     = tio.ScalarImage(args.samplePath)
    print(img.shape)
    myslice = []
    for i, s in enumerate(args.slice):
        if s >= 0: myslice.append(s)
        if s < 0: myslice.append(slice(0,img.shape[i],1))
    img     = img.data.numpy()[tuple(myslice)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,3)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto')
    fig.savefig(args.outPath, dpi=300)

if __name__ == '__main__':
    main()
