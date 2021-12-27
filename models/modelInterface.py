import torch
from . import networks
from torchsummary import summary
from collections import OrderedDict
from utils.util import getBaseMidApexImgs
import os 

class ModelInterface():
    """ Interface for model Unet3D"""
    
    def __init__(self, opt):
        """Initialize the model interface.

        Parameters:
            opt (Option class)-- stores all the experiment flags;
        """
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_names = ['train', 'val']
        self.visual_names = ['img', 'msk', 'pred']
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        # define network
        self.net = networks.Unet3D(1, 1, 7, opt.ngf)
        self.net.to(self.device)
        if opt.phase == "train":
            # define loss function
            if opt.loss == 'L1':
                self.criterion = torch.nn.L1Loss()   
            elif opt.loss == 'Dice':
                self.criterion = networks.DiceLoss()
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
        # if not self.isTrain or opt.continue_train:
        #     load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
        #     self.load_networks(load_suffix)
        
        #Print Network information
        try:
            summary(self.net, (1, opt.load_size_d, opt.load_size_h, opt.load_size_w))
        except:
            print(self.net)


    def get_current_visuals(self):
        """Return visualization images from 3D arrays. train.py will display these images with visdom, and save the images to a HTML"""
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


    def set_input(self, input):
        """Unpack input data from the dataloader
        Parameters:
            input (dict): include the data itself and its path name.
        """
        self.img = input['img'].to(self.device)
        self.msk = input['msk'].to(self.device)
        self.path = input['path']

    def val(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.loss_val = self.criterion(self.pred, self.msk) * self.opt.lambda_L1


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred = self.net(self.img)

    def backward(self):
        """Calculate loss"""
        self.loss_train = self.criterion(self.pred, self.msk) * self.opt.lambda_L1
        self.loss_train.backward()

    def optimize_parameters(self):
        #Ford
        self.forward()                   # compute predictions Net(img)
        # Backprop
        self.optimizer.zero_grad()        # set Net's gradients to zero
        self.backward()                   # calculate graidents
        self.optimizer.step()             # udpate Net's weights
