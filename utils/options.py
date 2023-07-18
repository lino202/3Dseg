import argparse


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Base Options")
        #Base
        self.parser.add_argument('--root_path',       required=True, type=str, help='path to data')       
        self.parser.add_argument('--name',            required=True, type=str, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--results_dir',     required=True, type=str, help='models are saved here')
        self.parser.add_argument('--phase',           required=True, type=str, help='train or test')
        #Model
        self.parser.add_argument('--input_nc',        type=int,      default=1,          help='# of input channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--output_nc',       type=int,      default=1,          help='# of output classes: 1 if binary and N for multi-class segmentation')
        self.parser.add_argument('--nfl',             type=int,      default=64,         help='# of filters in the last conv layer')
        self.parser.add_argument('--num_downs',       type=int,      default=7,          help='# of deep layers in the Unet architecture')
        self.parser.add_argument('--norm',            type=str,      default='instance', help='instance normalization or batch normalization [instance | batch ]')
        self.parser.add_argument('--no_dropout',      action='store_true',               help='no dropout')
        # Dataset 
        self.parser.add_argument('--batch_size',      type=int,      default=1,   help='input batch size')
        self.parser.add_argument('--patch_size',      type=int,      nargs='+',   help='patch size HeightxWidthxDepth', required=True)
        self.parser.add_argument('--display_winsize', type=int,      default=256, help='display window size for both visdom and HTML')


class TrainOptions():

    def __init__(self):
        self.parser = BaseOptions().parser
        # visdom and HTML visualization parameters       
        self.parser.add_argument('--no_html',         action='store_true',                  help='do not save intermediate training results to [opt.results_dir]/[opt.name]/web/')
        self.parser.add_argument('--display_ncols',   type=int, default=3,                  help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--display_id',      type=int, default=1,                  help='window id of the web display')
        self.parser.add_argument('--display_server',  type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_env',     type=str, default='main',             help='visdom display environment name (default is "main")')
        self.parser.add_argument('--display_port',    type=int, default=8097,               help='visdom port of the web display')
        self.parser.add_argument('--save_epoch_freq', type=int, default=300,                help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--print_iter',      type=int, default=10,                 help='frequency of showing iteration info')
        #self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')

        # network saving and loading parameters
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count',    type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')   
        # training parameters
        self.parser.add_argument('--n_epochs',       type=int,   default=15,       help='number of epochs with the initial learning rate')
        self.parser.add_argument('--n_epochs_decay', type=int,   default=15,       help='number of epochs to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1',          type=float, default=0.5,      help='momentum term of adam')
        self.parser.add_argument('--lambda_loss',    type=float, default=100.0,    help='weight for loss')
        self.parser.add_argument('--lr',             type=float, default=0.01,     help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy',      type=str,   default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        self.parser.add_argument('--lr_decay_iters', type=int,   default=50,       help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--loss',           type=str,   default='Dice',   help='loss function [Dice | L1 ]')
        self.parser.add_argument('--n_layers_D',     type=int,   default=3,        help='only used if netD==n_layers, if the patch size is anisotropic this drives the number of 2D convs')
        self.parser.add_argument('--gan_mode',       type=str,   default='lsgan',  help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        self.parser.add_argument('--gan',            action='store_true',          help='perform GAN training')
        self.parser.add_argument('--dataaug',        action='store_true',          help='use the data augmentation for volumes')

class TestOptions():

    def __init__(self):
        self.parser = BaseOptions().parser
        self.parser.add_argument('--load_filename',         type=str,             help='file name of trained parameters')
        self.parser.add_argument('--ph',                    action='store_true',  help='use topo loss for adjusting results')
        self.parser.add_argument('--priorName',             type=str,             help='prior as a global var must be defined in utils/priors.py')
        self.parser.add_argument('--phParallel',            action='store_true',  help='use parallel calculation of PH')
        self.parser.add_argument('--phConstruction',        type=str,             help='Either 0 (4 (2D) or 6 (3D) connectivity) or N (8 (2D) or 26 (3D) connectivity)', default='N')
        self.parser.add_argument('--phThres',               type=float,           help='threshold for cutting a roi in ph post-ptocessing')
