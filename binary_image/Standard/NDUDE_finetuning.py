
# coding: utf-8

# In[1]:


from core import *
from tools import *
import numpy as np
from numpy import *
import os
import tensorflow as tf
import keras as K

import sys
import argparse


# In[2]:


img_arr = ['barbara_512.png', 'boat_512.png', 'cman_256.png', 'couple_512.png', 'Einstein_256.jpeg', 'fruit_256.bmp', 'lena_512.jpg', 'pepers_256.png']

try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--t", help="PI type", type=int)
    parser.add_argument("--i", help="image number: 0~67", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    img_num = args.i
    k = 50
    
except:
    result_name = "test"
    type_num = 1
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    img_num = 5
    k = 50
PI_type_arr = ['20%', '30%', '10%']


# In[3]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.backend.set_session(session)


# In[4]:


PI_type = PI_type_arr[type_num]
PI = load_channel('true', nb_x_classes, 1, type_num)

print(PI)

f = open('results/' + '%d_%s_'%(nb_x_classes, PI_type) + '_' + result_name,'a')

param_name = "ICE_%d_%d"%(nb_x_classes, type_num)

x,z = load_img(PI, [img_arr[img_num]])

n = len(x)

print(x[:20])
print(z[:20])

print(n)

print(error_rate(x,z))


# In[5]:


Estimated_PI = load_PI(param_name)
print(Estimated_PI)


# In[6]:


# Denoising Estimation Process
Estimated_PI = load_PI(param_name)
SE = State_Estimation_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
error, x_hat = SE.N_DUDE(Estimated_PI)
f.write("%d %.5f\n"%(img_num, error))
print('%d %.5f'%(img_num, error))

