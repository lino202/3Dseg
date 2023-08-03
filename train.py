"""General-purpose training script """

import sys
import numpy as np
import time
import os
from collections import OrderedDict
from data import create_dataloader
from utils.options import TrainOptions
from models.model_unet import ModelUnet3D
from utils.visualizer import Visualizer
from utils.util import Logger, mkdir, saveLossPlot

def main():
    start = time.time()
    opt = TrainOptions().parser.parse_args()
    mkdir(os.path.join(opt.results_dir, opt.name))
    sys.stdout = Logger(os.path.join(opt.results_dir, opt.name, "train_output.out"))

    print('Reading data from {}'.format(opt.root_path))
    train_dataloader = create_dataloader.create(opt, opt.phase)     # create a dataloader with given options
    print('Training with {} samples grouped in {} batches'.format(len(train_dataloader.dataset),len(train_dataloader)))
    val_dataloader = create_dataloader.create(opt, 'val')  # create a create a dataloader with given options
    print('Validating with {} samples grouped in {} batches'.format(len(val_dataloader.dataset), len(val_dataloader)))

    model        = ModelUnet3D(opt)         # create a Model
    model.setup(opt)                        # regular setup: load and print networks; create schedulers
    visualizer   = Visualizer(opt)          # create a visualizer that display/save images and plots
    min_val_loss = np.inf

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        #Train-----------------------------------------------
        model.net.train()
        iteration = 0
        train_losses = []
        for i, data in enumerate(train_dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            train_losses.append(model.loss_train.item())
            iteration += 1
            if iteration % opt.print_iter == 0:
                print("Train iteration {:d} in {:.4f} s".format(iteration, time.time() - iter_start_time))

        train_visuals = model.get_current_visuals()
        visuals = OrderedDict()
        for k, v in train_visuals.items(): visuals[k+"_train"] = v 

        #Val--------------------------------------------------
        model.net.eval()
        iteration = 0
        val_losses = []
        for i, data in enumerate(val_dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.val()                     # calculate loss functions from validation dataset

            val_losses.append(model.loss_val.item())
            iteration += 1
            if iteration % opt.print_iter == 0:
                print("Validation iteration {:d} in {:.4f} s".format(iteration, time.time() - iter_start_time))


        # print training losses and save logging information to the disk
        losses = OrderedDict({'train': sum(train_losses)/len(train_losses), 'val': sum(val_losses)/len(val_losses)})
        visualizer.print_losses(epoch, losses)
        if opt.display_id > 0:
            visualizer.plot_current_losses(epoch, losses)

        # display images on visdom and save them on html
        val_visuals = model.get_current_visuals()
        for k, v in val_visuals.items(): visuals[k+"_val"] = v 
        visualizer.display_current_results(visuals, epoch, True)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('Saving the model at the end of epoch %d' % (epoch))
            model.save_network('latest')
            model.save_network(epoch)

        if min_val_loss > losses["val"]:
            print("Saving model with minor val loss in epoch {:d}".format(epoch))
            model.save_network('best_val_loss')
            min_val_loss = losses["val"]
            
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    print('Saving the model at the End of all epochs, epoch %d' % (epoch))
    model.save_network('latest')
    saveLossPlot(visualizer.vis, opt.results_dir, opt.name, opt.display_env, model.loss_names)

    print("Total training time was {} s".format(time.time() - start))

if __name__ == '__main__':
    main()
    
