
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


img_arr = ['2012_000003.jpg', '2012_000004.jpg', '2012_000007.jpg', '2012_000010.jpg', '2012_000014.jpg', '2012_000015.jpg', '2012_000016.jpg', '2012_000019.jpg', '2012_000025.jpg', '2012_000027.jpg', '2012_000028.jpg', '2012_000029.jpg', '2012_000030.jpg', '2012_000031.jpg', '2012_000032.jpg', '2012_000035.jpg', '2012_000036.jpg', '2012_000040.jpg', '2012_000042.jpg', '2012_000044.jpg', '2012_000045.jpg', '2012_000049.jpg', '2012_000050.jpg', '2012_000051.jpg', '2012_000055.jpg', '2012_000056.jpg', '2012_000058.jpg', '2012_000059.jpg', '2012_000060.jpg', '2012_000065.jpg', '2012_000067.jpg', '2012_000069.jpg', '2012_000070.jpg', '2012_000071.jpg', '2012_000072.jpg', '2012_000074.jpg', '2012_000078.jpg', '2012_000083.jpg', '2012_000084.jpg', '2012_000085.jpg', '2012_000086.jpg', '2012_000087.jpg', '2012_000089.jpg', '2012_000100.jpg', '2012_000102.jpg', '2012_000104.jpg', '2012_000105.jpg', '2012_000106.jpg', '2012_000108.jpg', '2012_000113.jpg']

try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--t", help="PI type", type=int)
    parser.add_argument("--o", help="Markov chain order", type=int)
    parser.add_argument("--i", help="image number: 0~67", type=int)
    
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    order = args.o
    img_num = args.i
    
except:
    result_name = "test"
    type_num = 0
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    img_num = 0
    order = 3
PI_type_arr = ['20%', '30%', '10%']


# In[3]:


PI_type = PI_type_arr[type_num]

PI = load_channel('true', nb_x_classes, 1, type_num)
assumed_PI = load_channel('assumed', nb_x_classes, order, type_num)
assumed_TRANS = load_TRANS('assumed', nb_x_classes, order)


f = open('results/' + '%d_%s_'%(nb_x_classes, PI_type) + '_' + result_name,'a')

param_name = "BW_%d_%d_%d"%(nb_x_classes, order, type_num)

x,z = load_img(PI, [img_arr[img_num]])
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
print(assumed_TRANS)


# In[4]:


BW_ = BW(n, x_conv, z)
a, Estimated_PI, gamma = BW_.Baum_Welch(assumed_TRANS, assumed_PI)
_, x_hat = BW_.denoise(gamma)
error = normalized_error_rate(x, x_hat%nb_x_classes, raw_error)
f.write("%d %d %.5f\n"%(order, img_num, error))
print(error)
print(Estimated_PI)

