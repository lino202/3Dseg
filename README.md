# Description

Unet 3D for cardiac segmentation.

## Prerequisites
- Linux or macOS or Win
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

- Clone this repo:
```bash
git clone https://github.com/lino202/3Dseg
cd 3Dseg
```

### Training

Explore data/create_dataset.py to get your data ready and before running init the visdom server
if you would like to see the training proccess with 

```bash
python -m visdom.server
```

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [topological-losses](https://github.com/nick-byrne/topological-losses).