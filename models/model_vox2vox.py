import torch
from . import networks
from . import customLosses
from torchsummary import summary
from collections import OrderedDict
from utils.util import getBaseMidApexImgs
import os 

class ModelVox2Vox():
    """ Interface for model Vox2Vox"""
    
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
        self.visual_names = ['real_A', 'real_B', 'fake_B']
        
        # define network
        self.netG = networks.Unet3D(opt.input_nc, opt.output_nc, opt.num_downs, opt.nfl)
        self.netG.to(self.device)
        if opt.phase == "train":
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
            self.model_names = ['G', 'D']
            
            self.netD = networks.NLayerDiscriminator(opt.input_nc + opt.output_nc)
            # self.netD = networks.NLayerDiscriminator(opt.input_nc)
            self.netD.to(self.device)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if opt.phase=="train":
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        elif opt.phase=="test":
            # load_filename = '%s_net_%s.pth' % (epoch, name) = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_network(opt.load_filename)
        else: raise ValueError("Wrong {} phase".format(opt.phase))
        
        # Print Network information
        try:
            summary(self.netG, (1, opt.load_size_d, opt.load_size_h, opt.load_size_w))
        except:
            print(self.netG)
        print(self.netD)


    def load_network(self, load_filename):
        """Load network from the disk.
        Parameters:
            load_filename (str) -- trained model file name
        """
        load_path = os.path.join(self.save_dir, "{}.pth".format(load_filename))
        print('Loading the model from {}'.format(load_path))
        state_dict = torch.load(load_path, map_location=self.device)
        self.netG.load_state_dict(state_dict)


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
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
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
        #As the dataloader is common for the normal training we need to change two things:
        #The real_A -> msk, realB -> img
        #Also the msk is not in the range [-1,1] as not preprocessing normalization is applied (implemented in customToTensor)
        self.real_A = input['msk'].to(self.device)
        self.real_B = input['img'].to(self.device)
        self.path   = input['path']
        self.affine = input['affine']

    # def val(self):
    #     """There's no validation for GANs
    #     """
    #     pass

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
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_loss
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

