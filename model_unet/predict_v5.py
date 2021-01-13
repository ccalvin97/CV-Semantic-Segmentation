# -*- coding: utf-8 -*-
#!/usr/bin/env python

import cv_unet_v5 as modelling
# from tools.data import test_generator, save_results
import sys
import os



path_pred = '/home/GDDC-CV-test1/Desktop/data/prediction_x_example_black'
path_pred_out = '/home/GDDC-CV-test1/Desktop/data/prediction_y'

PIXEL = 1200    #set your image size
BATCH_SIZE = 1
lr = 0.001
EPOCH = 1
X_CHANNEL = 3  # training data channel
Y_CHANNEL = 1  # test data channel
smooth = 1
normalisation = False




if __name__ == "__main__":
    """ Prediction Script
    Run this Python script with a command line
    argument that defines number of test samples
    e.g. python predict.py 6
    Note that test samples names should be:
    1.jpg, 2.jpg, 3.jpg ...
    """

    print('Start Prediction Programme')
    cur_dir=os.getcwd()
    paprameter_file = 'model_parameter'
    filename=os.listdir(cur_dir +'/'+ paprameter_file)
    model_weights_name = cur_dir + '/' + paprameter_file + '/'+ filename[0]
    
    
    network=modelling.network(PIXEL=PIXEL, BATCH_SIZE=BATCH_SIZE, lr=lr, EPOCH = EPOCH, X_CHANNEL = X_CHANNEL,
                              Y_CHANNEL = Y_CHANNEL, smooth = smooth, normalisation = False)

    # build model
    model = network.get_unet(pretrained_weights = model_weights_name)
    test_gen = network.test_generator(path_pred)
    images = os.listdir(path_pred)
    images.sort()
    results = model.predict_generator(test_gen,len(images),verbose=1)
    modelling.save_results(path_pred_out, results, images)

  
