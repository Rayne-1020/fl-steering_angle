from __future__ import print_function
#import pandas as pd
#import bagpy
#from bagpy import bagreader
from collections import defaultdict
from os import path
import numpy as np
from collections import defaultdict
import os
from PIL import Image
from keras.applications import vgg16
import keras

from timestr import *

def read_steerings(steering_log):
    steerings = defaultdict(list)
    for i in range(0,len(steering_log['Time'])):
        second = float(steering_log['Time'][i])
        angle = round(float(steering_log['steering_wheel_angle'][i]),4)
        timestamp = "%.1f" %second
        steerings[timestamp].append(angle)
    return steerings 

def read_image_stamps(image_log,dataset_path):
    timestamps = defaultdict(list)
    Timestr = get_timestr(dataset_path)
    for i in range(0,len(image_log['Time'])):
        second = float(image_log['Time'][i])
        timestamp = "%.1f" %second
        #image_id = "%.6f" %float(str(image_log['header.stamp.secs'][i]) + "." + str(image_log['header.stamp.nsecs'][i]))
        #image_id = "%.6f" %second
        image_id = Timestr[i]
        #print(image_id)
        timestamps[timestamp].append(image_id)
    return timestamps  

def read_images(image_folder, ids, image_size):
    imgs = []
    for id in ids:
        img_path = os.path.join(image_folder, '%s.png' %id) 
        with Image.open(img_path) as img:
            #resize img
            #(width, height) = (img.width*image_size, img.height*image_size)
            #img = img.resize((width, height))
            img = img.resize(image_size)
            #print(img.size)
        imgs.append(np.asarray(img))
    #print(imgs[0].shape)
    img_block = np.stack(imgs, axis = 0)
    #print(img_block.shape)#, img_block)
    if keras.backend.image_data_format() == 'th':
        print("yes")
        img_block = np.transpose(img_block, axes = (0, 3, 1, 2))
    return img_block

def vgg_preprocess_input(x):
    return vgg16.preprocess_input(x)
def normalize_input(x):
    return x / 255. 
def exact_output(y):
    return y
def normalize_output(y):
    return y / 5.

def data_generator(steering_log, dataset_path, image_log, image_folder, gen_type='train',
                   batch_size=32, image_size=0.5, timestamp_start=None, 
                   timestamp_end=None, shuffle=None, preprocess_input=normalize_input,
                   preprocess_output=exact_output):
    #setup
    minmax = lambda xs: (min(xs), max(xs))
    

    #read steering and image log
    steerings = read_steerings(steering_log) #dictionay with time stamps and the steerings within this time_stamps 4
    image_stamps = read_image_stamps(image_log,dataset_path) # dictionary with time stamps and the steerings within this time_stamps 2
    
    steerings_keys = list(steerings.keys())
    steerings_keys_float = list(map(float, steerings_keys))
    image_stamps_keys = list(image_stamps.keys())
    image_stamps_keys_float = list(map(float, image_stamps_keys))

    #statistics report
    #print('timestamp range for all steerings: %.1f, %.1f' %minmax(steerings_keys_float))
    #print('timestamp range for all image: %.1f, %.1f' %minmax(image_stamps_keys_float))
    #print('min and max # of steerings per time unit: %.1f, %.1f' %minmax(list(map(len, steerings.values()))))
    #print('min and max # of image per time unit: %.1f, %.1f' %minmax(list(map(len, image_stamps.values()))))

    #generate images and steerings within one time unit
    #mean steering will be used for multiple steering angles within the unit
    start = max(min(steerings_keys_float), min(image_stamps_keys_float))
    if timestamp_start:
        start = max(start, timestamp_start)
    end = min(max(steerings_keys_float), max(image_stamps_keys_float))
    if timestamp_end:
        end = min(end, timestamp_end)
    #print("sampling data from timestamp %.1f to %.1f" %(start,end))

    i = start
    x_buffer, y_buffer, buffer_size = [], [], 0
    #print(x_buffer, y_buffer, buffer_size)
    while True:
        if i > end:
            i = start
        #print(i,image_stamps[str(i)])
        if steerings[str(i)] and image_stamps[str(i)]:
            #find next barch of images and steering
            images = read_images(image_folder, image_stamps[str(i)], image_size) #image_stamp[i] == ids
            #print(images.shape)
            #mean angle within a time unit
            angles = np.repeat([np.mean(steerings[str(i)])], images.shape[0])
            x_buffer.append(images)
            y_buffer.append(angles)
            buffer_size += images.shape[0]
            #print(buffer_size, images.shape[0], batch_size)
            if buffer_size >= batch_size:
                indx = range(buffer_size)
                #np.random.shuffle(indx)
                x = np.concatenate(x_buffer, axis=0)[indx[:batch_size], ...]
                y = np.concatenate(y_buffer, axis=0)[indx[:batch_size], ...]
                x_buffer, y_buffer, buffer_size = [], [], 0
                yield preprocess_input(x.astype(np.float32)), preprocess_output(y)
        if shuffle:
            i = round(np.random.uniform(start,end),1)
            while str(i) not in image_stamps_keys:
                i = np.random.uniform((start,end),1)
        else:
            #without shuffle, based on order, it should move every 0.1
            i += 0.1
            i = round(i,1)
            # while str(i) not in image_stamps_keys:
            #     i += 0.1

