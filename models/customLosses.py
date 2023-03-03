import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.01):
        #flatten label and prediction tensors
        inputs = (inputs.view(-1) + 1) / 2
        targets = (targets.view(-1) + 1) / 2

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return 1 - dice