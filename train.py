"""General-purpose training script """

from collections import OrderedDict
import time
from data import create_dataset
from utils.options import TrainOptions
from models.modelInterface import ModelInterface
from utils.visualizer import Visualizer
import numpy as np

def main():
    
    opt = TrainOptions().parser.parse_args()
    train_dataset = create_dataset.create(opt, opt.phase)  # create a dataset given opt.dataset_mode and other options
    train_dataset_size = len(train_dataset)                # get the number of images in the dataset.
    print('The number of training volumes = {}'.format(train_dataset_size))

    val_dataset = create_dataset.create(opt, 'val')  # create a dataset given opt.dataset_mode and other options
    val_dataset_size = len(val_dataset)              # get the number of images in the dataset.
    print('The number of validation volumes = {}'.format(val_dataset_size))

    model        = ModelInterface(opt)      # create a Model Interface
    model.setup(opt)                        # regular setup: load and print networks; create schedulers
    visualizer   = Visualizer(opt)          # create a visualizer that display/save images and plots
    total_iters  = 0
    min_val_loss = np.inf

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs;
        epoch_start_time = time.time()  # timer for entire epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        #Train-----------------------------------------------
        model.net.train()
        iteration = 0
        train_losses = []
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            train_losses.append(model.loss_train.item())
            iteration += 1
            print("Train iteration {:d} in {:.4f} s".format(iteration, time.time() - iter_start_time))

        train_visuals = model.get_current_visuals()
        visuals = OrderedDict()
        for k, v in train_visuals.items(): visuals[k+"_train"] = v 

        #Val--------------------------------------------------
        iteration = 0
        model.net.eval()
        val_losses = []
        for i, data in enumerate(val_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.val()                     # calculate loss functions from validation dataset

            val_losses.append(model.loss_val.item())
            iteration += 1
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

if __name__ == '__main__':
    main()