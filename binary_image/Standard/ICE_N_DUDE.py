
# coding: utf-8

# In[ ]:


from core import *
from tools import *
import numpy as np
from numpy import *
import os
import tensorflow as tf
import keras as K

import sys
import argparse


# In[ ]:


img_arr = ['barbara_512.png', 'boat_512.png', 'cman_256.png', 'couple_512.png', 'Einstein_256.jpeg', 'fruit_256.bmp', 'lena_512.jpg', 'pepers_256.png']

try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--t", help="PI type", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    k = 50
    
except:
    result_name = "test"
    type_num = 1
    nb_x_classes = 8
    nb_z_classes = nb_x_classes
    assumed_delta = 0.4
    k = 50
PI_type_arr = ['20%', '30%', '10%']


# In[ ]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.backend.set_session(session)


# In[ ]:


PI_type = PI_type_arr[type_num]
PI = load_channel('true', nb_x_classes, 1, type_num)
assumed_PI = load_channel('assumed', nb_x_classes, 1, type_num)

print(PI)
print(assumed_PI)


x,z = load_img(PI, img_arr)

x, z = load_img(PI, img_arr[:10])

param_name = "ICE_%d_%d"%(nb_x_classes, type_num)

n = len(x)

print(x[:20])
print(z[:20])

print(n)

print(error_rate(x,z))


# In[ ]:


# Parameter Estimation Process
ICE_N_DUDE = ICE_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
Estimated_PI = ICE_N_DUDE.ICE(assumed_PI)
print(Estimated_PI)

