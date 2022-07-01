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
import tensorflow as tf
from Data_Generator import *
from Model import *
from tensorflow.keras import backend as K
from keras.callbacks import ModelCheckpoint

def scale_model_weights(weight, scaler):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scaler*weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    ''' return the sum of the listed scaled weights. Equivalent to scaled average of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.reduce_sum(grad_list_tuple, axis = 0)
        avg_grad.append(layer_mean)
    return avg_grad

def main():
    parser = argparse.ArgumentParser(description="Testing Udacity SDC data")
    parser.add_argument('--communication_round', type = int, help = 'communicational rounds of federated learning framework') 
    args = parser.parse_args()
    comms_round = args.communication_round
    #initialize a global model using a federated learning round
        #global_model = ... #how to import model
    #get local datasets
    # client_dataset_paths = ['/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_1',
    #                         '/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_2',
    #                         '/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_4',
    #                         '/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_6']
    test_dataset_path = '/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_4'
    client_dataset_samples = [4401,15796,4235,7402]
    clients = {'client1': {'client_dataset_paths':'/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_1', 'client_dataset_samples': 4401},
           'client2': {'client_dataset_paths':'/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_2', 'client_dataset_samples': 15796},
           'client3': {'client_dataset_paths':'/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_5', 'client_dataset_samples': 4235},
           'client4': {'client_dataset_paths':'/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_6', 'client_dataset_samples': 7402}}
    global_count = np.sum(client_dataset_samples)
    # clients = {}
    # for i in range(0,4):
    #     clients[i]
    model_name = 'cnn1' 
    #model = 'cnn2'
    image_size = (480,320)
    batch_size = 32
    nb_epoch = 1 #every communication round, just do 1 epoch

    model_builders = {'cnn1': (build_cnn1, normalize_input, exact_output),'cnn2': (build_cnn2, normalize_input, exact_output)}
    if model_name not in model_builders:
        raise ValueError("unsupported model %s" %model_name)
    model_builder, input_processor, output_processor = model_builders[model_name]
    #build global model 
    global_model = model_builder(image_size)
    print('global model %s built...' %model_name)
    global_model.summary()
    Test_RMSE_local1 = []
    Test_RMSE_local2 = []
    Test_RMSE_local3 = []
    Test_RMSE_local4 = []
    Test_RMSE_global = []
    for comm_round in range(comms_round):
        #get global model weight
        #global_weights = global_model.get_weights()
        #collect local model weights after scalling
        global_count = np.sum(client_dataset_samples)
    
        #initialize a list
        scaled_local_weight_list = list()
        #get performance of all the local models
        client_model = {}
        client_model['client1'] = {}
        client_model['client2'] = {}
        client_model['client3'] = {}
        client_model['client4'] = {}
        if comm_round == 0:#the first comm_round, generate first global model
            #loop through each client and create new local model
            for client in clients:
                #first get the dataset from each client
                steering_log_path = path.join(clients[client]['client_dataset_paths'], 'vehicle-steering_report.csv')
                image_log_path = path.join(clients[client]['client_dataset_paths'],'center_camera-image_color-compressed.csv')
                camera_images = path.join(clients[client]['client_dataset_paths'], 'images')
                steering_log = pd.read_csv(steering_log_path)
                image_log = pd.read_csv(image_log_path)
                #then build model for each client
                local_model = model_builder(image_size)
                print('local model %s built...' %model_name)
                #local_model.summary()

                train_generator = data_generator(steering_log = steering_log,
                                        dataset_path = clients[client]['client_dataset_paths'] + '.bag',
                                        image_log = image_log,
                                        image_folder = camera_images,
                                        gen_type='train',
                                        batch_size = batch_size,
                                        image_size = image_size,
                                        timestamp_start = None,
                                        timestamp_end = None,#full dataset train
                                        shuffle = True,
                                        preprocess_input = input_processor,
                                        preprocess_output = output_processor)

                local_model.fit(train_generator, epochs = nb_epoch, steps_per_epoch = round(clients[client]['client_dataset_samples']/batch_size), verbose = 1)
                #scale local_model weights and add to the list
                scaling_factor = clients[client]['client_dataset_samples']/global_count
                scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
                client_model[client]['model'] = local_model
                K.clear_session()

                
            #update the global model after the first communication round
            average_weights = sum_scaled_weights(scaled_local_weight_list)
            global_model.set_weights(average_weights)
            global_weights = global_model.get_weights()

        if comm_round > 0:#the rest of the comm_round, we need to update the global model weight based on precious local models
            #loop through each client and create new local model
            for client in clients:
                #first get the dataset from each client
                steering_log_path = path.join(clients[client]['client_dataset_paths'], 'vehicle-steering_report.csv')
                image_log_path = path.join(clients[client]['client_dataset_paths'],'center_camera-image_color-compressed.csv')
                camera_images = path.join(clients[client]['client_dataset_paths'], 'images')
                steering_log = pd.read_csv(steering_log_path)
                image_log = pd.read_csv(image_log_path)
                #then build model for each client
                model_builders = {'cnn1': (build_cnn1, normalize_input, exact_output),'cnn2': (build_cnn2, normalize_input, exact_output)}
                if model_name not in model_builders:
                    raise ValueError("unsupported model %s" %model_name)
                model_builder, input_processor, output_processor = model_builders[model_name]
                local_model = model_builder(image_size)
                print('local model %s built...' %model_name)
                #local_model.summary()
                train_generator = data_generator(steering_log = steering_log,
                                        dataset_path = clients[client]['client_dataset_paths'] + '.bag',
                                        image_log = image_log,
                                        image_folder = camera_images,
                                        gen_type='train',
                                        batch_size = batch_size,
                                        image_size = image_size,
                                        timestamp_start = None,
                                        timestamp_end = None,#full dataset train
                                        shuffle = True,
                                        preprocess_input = input_processor,
                                        preprocess_output = output_processor)
                #set local model weight to the weight of the global model
                local_model.set_weights(global_weights)
                history = local_model.fit(train_generator, epochs = nb_epoch, steps_per_epoch = round(clients[client]['client_dataset_samples']/batch_size), verbose = 1)
                #scale local_model weights and add to the list
                #print(history.history['loss'])
                #print(history.history['loss'])
                if history.history['loss'][0] <= 0.001:
                    #performance is good enough, no need to update 
                    local_count = 0
                    global_count -= clients[client]['client_dataset_samples']
                    print(client,"will not update the global model")
                else:
                    global_count = global_count
                    local_count = clients[client]['client_dataset_samples']
                #scaling_factor = local_count/global_count
                #print(scaling_factor)
                #scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
                #scaled_local_weight_list.append(scaled_weights)
                client_model[client]['model'] = local_model
                client_model[client]['scaled_counts'] = local_count
                K.clear_session()
            print(global_count)#UPDATED GLOBAL COUNTS
            for client in clients:
                scaling_factor = client_model[client]['scaled_counts']/global_count
                scaled_weights = scale_model_weights(client_model[client]['model'].get_weights(),scaling_factor)
                scaled_local_weight_list.append(scaled_weights)
            #update the global model after the first communication round
            average_weights = sum_scaled_weights(scaled_local_weight_list)
            global_model.set_weights(average_weights)    
            global_weights = global_model.get_weights()
        print('communication round',comm_round,'global model successfully trained...')
        
        #test the trained global model on local dataset HMB_4
        #input the test steering_log, image_log...
        #or after communication rounds, save the model and then test
        test_steering_log_path = path.join(test_dataset_path, 'vehicle-steering_report.csv')
        test_image_log_path = path.join(test_dataset_path,'center_camera-image_color-compressed.csv')
        test_camera_images = path.join(test_dataset_path, 'images')
        test_steering_log = pd.read_csv(test_steering_log_path)
        test_image_log = pd.read_csv(test_image_log_path)
        test_generator = data_generator(steering_log = test_steering_log,
                                        dataset_path = test_dataset_path + '.bag',
                                        image_log = test_image_log,
                                        image_folder = test_camera_images,
                                        gen_type = 'test',
                                        batch_size = 1947,#the batch size of the test model dataset?
                                        image_size = image_size,
                                        timestamp_start= None,
                                        timestamp_end= None,
                                        shuffle = False,
                                        preprocess_input = input_processor,
                                        preprocess_output = output_processor)
        test_x, test_y = next(test_generator)
        #print('test data shape:', test_x.shape, test_y.shape)
        #test_x_change = np.transpose(test_x, axes = (0, 2, 1, 3))
        #print('test data shape changed:', test_x_change.shape, test_y.shape)
        yhat_global = global_model.predict(test_x)
        yhat_local1 = client_model['client1']['model'].predict(test_x)
        yhat_local2 = client_model['client2']['model'].predict(test_x)
        yhat_local3 = client_model['client3']['model'].predict(test_x)
        yhat_local4 = client_model['client4']['model'].predict(test_x)
        #accuracy = np.mean(abs(test_y - yhat/test_y))
        #print(yhat_global,test_y)
        mse_local1 = np.mean((yhat_local1 - test_y)**2)
        mse_local2 = np.mean((yhat_local2 - test_y)**2)
        mse_local3 = np.mean((yhat_local3 - test_y)**2)
        mse_local4 = np.mean((yhat_local4 - test_y)**2)
        mse_global = np.mean((yhat_global - test_y)**2)
        Test_RMSE_global.append(mse_global)
        Test_RMSE_local1.append(mse_local1)
        Test_RMSE_local2.append(mse_local2)
        Test_RMSE_local3.append(mse_local3)
        Test_RMSE_local4.append(mse_local4)
        print("RMSE using local model1:", Test_RMSE_local1)
        print("RMSE using local model2:", Test_RMSE_local2)
        print("RMSE using local model3:", Test_RMSE_local3)
        print("RMSE using local model4:", Test_RMSE_local4)
        print("RMSE in each comm_round:", Test_RMSE_global)
        #print("model test accuracy:", accuracy)
        # fileobj1 = open('/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/history/local1.txt','w')
        # for element in Test_RMSE_local1:
        #     fileobj1.write(element+",")
        # fileobj1.close()
        # fileobj2 = open('/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/history/local2.txt','w')
        # for element in Test_RMSE_local2:
        #     fileobj2.write(element+",")
        # fileobj2.close()
        # fileobj3 = open('/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/history/local3.txt','w')
        # for element in Test_RMSE_local3:
        #     fileobj3.write(element+",")
        # fileobj3.close()
        # fileobj4 = open('/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/history/local4.txt','w')
        # for element in Test_RMSE_local4:
        #     fileobj4.write(element+",")
        # fileobj4.close()
        # fileobj5 = open('/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/history/global.txt','w')
        # for element in Test_RMSE_global:
        #     fileobj5.write(element+",")
        # fileobj5.close()
    print("federated framework successfully trained")
    #save the trained global model
    client_model['client1']['model'].save("/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/models/local_model_1.hdf5")
    client_model['client2']['model'].save("/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/models/local_model_2.hdf5")
    client_model['client3']['model'].save("/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/models/local_model_3.hdf5")
    client_model['client4']['model'].save("/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/models/local_model_4.hdf5")
    global_model.save("/home/ai/Desktop/Rayne/research-FL-Rayne/steering_angle_prediction/models/global_model_practice_user_selection_cnn2_round_to_4.hdf5")
        
    # )
if __name__ == '__main__':
    main()
            
        
            

