# from __future__ import print_function, division
# import pandas as pd
# import matplotlib.pyplot as plt 
# from skimage import io, transform
# import os 
# import numpy as numpy

# import torch 
# from torch.utils.data import Dataset, DataLoader, Sampler
# from torchvision import transforms, utils
# import random

# class UdacityDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform = None, slice_frames = None, select_ratio = 1.0, select_range = None):
        
#         assert select_ratio >= -1.0 and select_ratio <= 1.0
        
#         image_csv = pd.read_csv(csv_file) # combined csv with image and steering angles
#         #no select camera, only central camera is used
#         csv_len = len(image_csv)
#         if slice_frames:
#             csv_selected = image_csv[0:0] # empty dataframe initial
#             for start_idx in range(0, csv_len, slice_frames): # from 0 to csv_len, with step = slice_frames
#                 if select_ratio > 0: # choose a portion from slice_frames, change the stepsize
#                     end_idx = int(start_idx + slice_frames * select_ratio)# select left
#                 else:
#                     start_idx, end_idx = int(start_idx + slice_frames * (1 + select_ratio)), start_idx + slice_frames
#                     #select_ratio = 0, select nothing
#                     #select_ratio < 0, select right  
#                 if end_idx > csv_len: 
#                     end_idx = csv_len
#                 if start_idx > csv_len: 
#                     start_idx = csv_len
#                 csv_selected = csv_selected.append(image_csv[start_idx:end_idx])
#             self.image_csv = csv_selected
#         elif select_range:
#             csv_selected = image_csv.iloc[select_range[0]:select_range[1]]
#             self.image_csv = csv_selected
#         else:
#             self.image_csv = image_csv
        
#         self.root = root_dir
#         self.transform = transform

#         #mean and cov value of steering angle
#         self.mean = np.mean(image_csv['steering_wheel_angle'])
#         self.std = np.std(image_csv['steering_wheel_angle'])

#     def __len__(self):
#         return len(self.image_csv)
    
#     def read_data_single(self, idx):
#         path = os.path.join(self.root_dir, self.image_csv)

        

