import torch
from . import networks
from . import customLosses
from torchsummary import summary
from collections import OrderedDict
from utils.util import getBaseMidApexImgs
import os 

class ModelUnet3D():
    """ Interface for model Unet3D"""
    
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
        self.visual_names = ['img', 'msk', 'pred']
        
        # define network
        self.net = networks.Unet3D(opt.input_nc, opt.output_nc, opt.num_downs, opt.nfl)
        self.net.to(self.device)
        if opt.phase == "train":
            self.loss_names = ['train', 'val']
            # define loss function
            if opt.loss == 'L1' and opt.output_nc == 1:
                self.criterion = torch.nn.L1Loss()
            elif opt.loss == 'CE':
                self.criterion = torch.nn.CrossEntropyLoss()
            elif opt.loss == 'Dice' and opt.output_nc == 1:
                self.criterion = customLosses.BinaryDiceLoss()
            else:
                raise ValueError("Loss is not defined") 

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))            
            # self.optimizers.append(self.optimizer)


    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if opt.phase=="train":
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
        elif opt.phase=="test":
            # load_filename = '%s_net_%s.pth' % (epoch, name) = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_network(opt.load_filename)
        else: raise ValueError("Wrong {} phase".format(opt.phase))
        
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


    def get_current_visuals(self):
        """Return visualization images from 3D arrays. 
        train.py will display these images with visdom, and save the images to a HTML
        test.py  will save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                imgsDict = getBaseMidApexImgs(getattr(self, name), name)
                for name in imgsDict.keys():
                    visual_ret[name] = imgsDict[name]
        return visual_ret

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizer.param_groups[0]['lr']
        
        if self.opt.lr_policy == 'plateau':
            self.scheduler.step(self.metric)
        else:
            self.scheduler.step()

        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def save_network(self, epoch):
        """Save network to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_filename = '%s_net.pth' % (epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        net = self.net
        torch.save(net.state_dict(), save_path)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    #TODO here only train, val or test are admitted 
    #when we want to test a new example a dummy, really simple script 
    #should be used.
    def set_input(self, input): 
        """Unpack input data from the dataloader
        Parameters:
            input (dict): include the data itself and its path name.
        """
        self.img    = input['img'].to(self.device)
        self.msk    = input['msk'].to(self.device)
        self.path   = input['path']
        self.affine = input['affine']

    def val(self):
        """Forward function used in val time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()
            self.loss_val = self.criterion(self.pred, self.msk) * self.opt.lambda_loss

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad(): 
            self.forward()
            # self.compute_visuals()


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred = self.net(self.img)

    def backward(self):
        """Calculate loss"""
        self.loss_train = self.criterion(self.pred, self.msk) * self.opt.lambda_loss
        self.loss_train.backward()

    def optimize_parameters(self):
        #Ford
        self.forward()                   # compute predictions Net(img)
        # Backprop
        self.optimizer.zero_grad()        # set Net's gradients to zero
        self.backward()                   # calculate gradients
        self.optimizer.step()             # udpate Net's weights
