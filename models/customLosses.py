import torch.nn as nn

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.01):
        #flatten label and prediction tensors
        #As this is a binary loss and last layer is tanh we adjust inputs in [0,1] range
        #and targets should be already in that range
        inputs = (inputs.view(-1) + 1) / 2
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return 1 - dice