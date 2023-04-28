"""General-purpose training script """

from collections import OrderedDict
import time
from data import create_dataloader
from utils.options import TrainOptions
from models import model_vox2vox
from utils.visualizer import Visualizer
import numpy as np

def main():
    
    opt = TrainOptions().parser.parse_args()
    train_dataloader = create_dataloader.create(opt, opt.phase)     # create a dataloader with given options
    print('Training with {} samples grouped in {} batches'.format(len(train_dataloader.dataset),len(train_dataloader)))   
    if not opt.gan: raise ValueError("Using the train GAN script with no gan option")

    model        = model_vox2vox.ModelVox2Vox(opt)      # create a Model
    model.setup(opt)                        # regular setup: load and print networks; create schedulers
    visualizer   = Visualizer(opt)          # create a visualizer that display/save images and plots
    total_iters  = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        #Train-----------------------------------------------
        model.netG.train()
        model.netD.train()
        iteration = 0
        train_losses = []
        for i, data in enumerate(train_dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            train_losses.append(model.get_current_losses())
            iteration += 1
            if iteration % opt.print_iter == 0:
                print("Train iteration {:d} in {:.4f} s".format(iteration, time.time() - iter_start_time))

        # display images on visdom and save them on html
        train_visuals = model.get_current_visuals()
        visuals = OrderedDict()
        for k, v in train_visuals.items(): visuals[k+"_train"] = v     
        visualizer.display_current_results(visuals, epoch, True)

        # print training losses and save logging information to the disk
        # get the mean of the losses computed in the epoch
        losses = OrderedDict()
        for loss_name in model.loss_names:
            tmp = []
            for train_loss in train_losses:
                tmp.append(train_loss[loss_name])
            losses["{}_mean".format(loss_name)] = sum(tmp)/len(tmp)
        visualizer.print_losses(epoch, losses)
        if opt.display_id > 0:
            visualizer.plot_current_losses(epoch, losses)


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('Saving the model at the end of epoch %d' % (epoch))
            model.save_networks('latest')
            model.save_networks(epoch)
            
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    print('Saving the model at the end of all epochs, epoch %d' % (epoch))
    model.save_networks('latest')

if __name__ == '__main__':
    start = time.time()
    main()
    print("Total training time was {} s".format(time.time() - start))