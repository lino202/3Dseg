import torch
from . import networks
from torchsummary import summary
from collections import OrderedDict
from utils.util import getBaseMidApexImgs
import os 

class ModelTest():
    """ Interface for model Unet3D in test"""
    
    def __init__(self, opt):
        """Initialize the model interface.

        Parameters:
            opt (Option class)-- stores all the experiment flags;
        """
        self.opt = opt
        if torch.cuda.is_available():
            print("{:s} is going to be use as device".format(torch.cuda.get_device_name(torch.cuda.current_device())))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = os.path.join(opt.results_dir, opt.name)
        self.visual_names = ['img', 'pred']
        
        # define network
        self.net = networks.Unet3D(opt.input_nc, opt.output_nc, opt.num_downs, opt.nfl)
        self.net.to(self.device)

    def setup(self, opt):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.load_network(opt.load_filename)
        
        # Print Network information
        try:
            summary(self.net, (1, opt.load_size_d, opt.load_size_h, opt.load_size_w))
        except:
            print(self.net)


    def load_network(self, load_filename):
        """Load network from the disk.
        Parameters:
            load_filename (str) -- trained model file name
        """
        load_path = os.path.join(self.save_dir, "{}.pth".format(load_filename))
        print('Loading the model from {}'.format(load_path))
        state_dict = torch.load(load_path, map_location=self.device)
        self.net.load_state_dict(state_dict)


    # def get_current_visuals(self):
    #     """Return visualization images from 3D arrays. 
    #     train.py will display these images with visdom, and save the images to a HTML
    #     test.py  will save the images to a HTML"""
    #     visual_ret = OrderedDict()
    #     for name in self.visual_names:
    #         if isinstance(name, str):
    #             imgsDict = getBaseMidApexImgs(getattr(self, name), name)
    #             for name in imgsDict.keys():
    #                 visual_ret[name] = imgsDict[name]
    #     return visual_ret

    def set_input(self, input): 
        """Unpack input data from the dataloader
        Parameters:
            input (dict): include the data itself and its path name.
        """
        self.img    = input['img'].to(self.device)
        self.path   = input['path']
        self.affine = input['affine']

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad(): 
            self.forward()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred = self.net(self.img)
