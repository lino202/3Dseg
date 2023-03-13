import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class Unet3D(nn.Module):
    """Create a Unet 3D"""

    def __init__(self, input_nc, output_nc, num_downs, nfl=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            nfl (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Unet3D, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(nfl * 8, nfl * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        
        for i in range(num_downs - 5):          # add intermediate layers with nfl * 8 filters
            unet_block = UnetSkipConnectionBlock(nfl * 8, nfl * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        
        # gradually reduce the number of filters from nfl * 8 to nfl
        unet_block = UnetSkipConnectionBlock(nfl * 4,   nfl * 8, input_nc=None,     submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nfl * 2,   nfl * 4, input_nc=None,     submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nfl,       nfl * 2, input_nc=None,     submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, nfl,     input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu   = nn.ReLU(True)
        upnorm   = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,kernel_size=4, stride=2,padding=1)
            down   = [downconv]
            up     = [uprelu, upconv, nn.Tanh()]
            model  = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,kernel_size=4, stride=2,padding=1, bias=use_bias)
            down   = [downrelu, downconv]
            up     = [uprelu, upconv, upnorm]
            model  = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,kernel_size=4, stride=2,padding=1, bias=use_bias)
            down   = [downrelu, downconv, downnorm]
            up     = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)



##############################################################################
# Maybe Useful
##############################################################################

# def get_norm_layer(norm_type='instance'):
#     """Return a normalization layer

#     Parameters:
#         norm_type (str) -- the name of the normalization layer: batch | instance | none

#     For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
#     For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
#     """
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
#     # elif norm_type == 'none':
#     #     def norm_layer(x): return Identity()
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer


# def init_weights(net, init_type='normal', init_gain=0.02):
#     """Initialize network weights.

#     Parameters:
#         net (network)   -- network to be initialized
#         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

#     We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
#     work better for some applications. Feel free to try yourself.
#     """
#     def init_func(m):  # define the initialization function
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             init.normal_(m.weight.data, 1.0, init_gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>


# def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
#     Parameters:
#         net (network)      -- the network to be initialized
#         init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         gain (float)       -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

#     Return an initialized network.
#     """
#     if len(gpu_ids) > 0:
#         assert(torch.cuda.is_available())
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
#     init_weights(net, init_type, init_gain=init_gain)
#     return net