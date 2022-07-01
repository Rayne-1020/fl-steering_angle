from __future__ import print_function

import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt 

import argparse
import numpy as np 
import pandas as pd
import os
import time 
from os import path

from Data_Generator import *
from Model import *

from keras.callbacks import ModelCheckpoint

def main():
    #parse argument
    parser = argparse.ArgumentParser(description="Testing Udacity SDC data")
    parser.add_argument('--dataset', type = str, help = 'dataset folder with csv and image folders') #/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_1
    parser.add_argument('--model', type = str, help = 'model to evaluate, current list: {cnn1, cnn2}')
    parser.add_argument('--resized_image_width', type=int, help='resized image width')
    parser.add_argument('--resized_image_height', type = int, help = 'resized image height')
    parser.add_argument('--nb-epoch', type=int, help='# of training epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    args = parser.parse_args()

    dataset_path = args.dataset
    model_name = args.model
    image_size = (args.resized_image_width, args.resized_image_height)
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch

    #build model and train it
    steering_log_path = path.join(dataset_path, 'vehicle-steering_report.csv')
    image_log_path = path.join(dataset_path,'center_camera-image_color-compressed.csv')
    camera_images = path.join(dataset_path, 'images')
    steering_log = pd.read_csv(steering_log_path)
    image_log = pd.read_csv(image_log_path)


    model_builders = {'cnn1': (build_cnn1, normalize_input, exact_output),'cnn2': (build_cnn2, normalize_input, exact_output)}

    if model_name not in model_builders:
        raise ValueError("unsupported model %s" %model_name)
    model_builder, input_processor, output_processor = model_builders[model_name]
    model = model_builder(image_size)
    print('model %s built...' %model_name)


    # train_generator = data_generator(steering_log = steering_log,
    #                                 image_log = image_log,
    #                                 image_folder = camera_images,
    #                                 gen_type='train',
    #                                 batch_size = batch_size,
    #                                 image_size = image_size,
    #                                 timestamp_start = None,
    #                                 timestamp_end = 1479424391.9,
    #                                 shuffle = True,
    #                                 preprocess_input = input_processor,
    #                                 preprocess_output = output_processor)

    
    model_saver = ModelCheckpoint(filepath = "/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/models/" + "%s_weights.hdf5" %model_name, verbose = 1, save_best_only=False)#, monitor = "val_loss")
    #model.fit(train_generator, validation_data = val_generator, epochs = nb_epoch, steps_per_epoch = 138, callbacks = [model_saver], verbose = 1)
    model.fit(train_generator, epochs = nb_epoch, steps_per_epoch = 138, callbacks = [model_saver], verbose = 1)
    print('model successfully trained.....')

    #test on the last 20% data
    # test_generator = data_generator(steering_log = steering_log,
    #                                 image_log = image_log,
    #                                 image_folder = camera_images,
    #                                 batch_size = 880,
    #                                 image_size = image_size,
    #                                 timestamp_start= 1479424391.9,
    #                                 timestamp_end= None,
    #                                 shuffle = False,
    #                                 preprocess_input = input_processor,
    #                                 preprocess_output = output_processor)
    
    test_x, test_y = next(test_generator)
    print('test data shape:', test_x.shape, test_y.shape)
    test_x_change = np.transpose(test_x, axes = (0, 2, 1, 3))
    print('test data shape changed:', test_x_change.shape, test_y.shape)
    yhat = model.predict(test_x_change)
    accuracy = np.mean(abs(test_y - yhat/test_y))
    rmse = np.sqrt(np.mean((yhat-test_y)**2))
    print("model evaluated RMSE:", rmse)
    print("model test accuracy:", accuracy)



                          
    # )
if __name__ == '__main__':
    main()