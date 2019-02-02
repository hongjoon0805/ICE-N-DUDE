
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
    parser.add_argument("--o", help="Markov chain order", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    order = args.o
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    
    
except:
    result_name = "test"
    type_num = 1
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    img_num = 0
    order = 3
PI_type_arr = ['20%', '30%', '10%']


# In[ ]:


PI_type = PI_type_arr[type_num]

PI = load_channel('true', nb_x_classes, 1, type_num)
assumed_PI = load_channel('assumed', nb_x_classes, order, type_num)
assumed_TRANS = load_TRANS('assumed', nb_x_classes, order)


f = open('results/' + '%d_%s'%(nb_x_classes, PI_type) + '_' + result_name,'a')

param_name = "BW_%d_%d_%d"%(nb_x_classes, order, type_num)

x,z = load_img(PI, img_arr)
n = len(x)

print(PI)
print(x[:20])
print(z[:20])
print(n)

x_conv = convert_sequence(x, order, nb_x_classes)

raw_error = error_rate(x,z)

print(raw_error)

print(x_conv[:20])
print(z[:20])


# In[ ]:


if order == 1:
    BW_ = BW(n, x_conv, z, param_name)
    a, Estimated_PI, gamma = BW_.Baum_Welch(assumed_TRANS, assumed_PI)
    print(Estimated_PI)
elif order == 2:
    BW_ = BW_2nd_channel(n, x_conv, z, param_name)
    a, Estimated_PI, gamma = BW_.Baum_Welch(assumed_TRANS, assumed_PI)
    print(Estimated_PI)
elif order == 3:
    BW_ = BW_3rd_channel(n, x_conv, z, param_name)
    a, Estimated_PI, gamma = BW_.Baum_Welch(assumed_TRANS, assumed_PI)
    print(Estimated_PI)

