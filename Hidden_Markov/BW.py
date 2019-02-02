
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


try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--t", help="PI type", type=int)
    parser.add_argument("--x", help="the number of x classes", type=int)
    parser.add_argument("--o", help="Markov chain order", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    nb_x_classes = args.x
    nb_z_classes = nb_x_classes
    order = args.o
    n = int(2e6)
    
except:
    result_name = "test"
    type_num = 0
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    order = 1
    n = int(2e6)
PI_type_arr = ['20%', '30%']


# In[3]:


PI_type = PI_type_arr[type_num]

PI = load_channel('true', nb_x_classes, 2, type_num)
assumed_PI = load_channel('assumed', nb_x_classes, order, type_num)
TRANS = load_TRANS('true', nb_x_classes, 2)
assumed_TRANS = load_TRANS('assumed', nb_x_classes, order)



f = open('results/' + '%d_'%nb_x_classes + PI_type + '_' + result_name,'a')

param_name = "BW_%d_%d_%d"%(nb_x_classes, order, type_num)

x_raw, z_raw = Hidden_Markov(n, TRANS, PI)

print(x_raw[:20])
print(z_raw[:20])

x, z = x_raw%nb_x_classes, z_raw%nb_x_classes

print(x[:20])
print(z[:20])

x_conv = convert_sequence(x, order, nb_x_classes)

raw_error = error_rate(x,z)

print(raw_error)

print(x_conv[:20])
print(z[:20])

print(PI)
print(assumed_PI)


# In[ ]:


BW_ = BW(n, x_conv, z)
a, Estimated_PI, gamma = BW_.Baum_Welch(assumed_TRANS, assumed_PI)
_, x_hat = BW_.denoise(gamma)
error = normalized_error_rate(x, x_hat%nb_x_classes, raw_error)
f.write("%d %.5f\n"%(order, error))
save_PI(Estimated_PI, param_name)
print(error)
print(Estimated_PI)
print(a)

