import roslib
import rosbag
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
import os
import sys
import pandas as pd 
def get_timestr(dataset_path):
    Timestr = [] 
    with rosbag.Bag(dataset_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics = ["/center_camera/image_color/compressed"]):
            #print("receive image of type:", msg.format)
            #np_arr = np.fromstring(msg.data,np.uint8)
            #cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            img_file = "/home/ai/Desktop/Rayne/research-FL-Rayne/Udacitydataset/Ch2_002/HMB_1/images"
            timestr = "%.6f" %msg.header.stamp.to_sec()
            Timestr.append(timestr)
    return Timestr

