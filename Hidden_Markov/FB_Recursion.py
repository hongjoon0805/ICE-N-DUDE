
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
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    nb_x_classes = args.x
    nb_z_classes = nb_x_classes
    n = int(1e6)
    
except:
    result_name = "test"
    type_num = 1
    nb_x_classes = 4
    nb_z_classes = nb_x_classes
    n = int(1e6)
PI_type_arr = ['20%', '30%']
delta_arr = [0.2, 0.3]


# In[3]:


PI_type = PI_type_arr[type_num]
PI = sym_mat(nb_x_classes, delta_arr[type_num])
TRANS = sym_mat(nb_x_classes, 0.1)

print(PI)

f = open('results/' + '%d_'%nb_x_classes + PI_type + '_' + result_name,'a')

x, z = Hidden_Markov(n, TRANS, PI)

raw_error = error_rate(x,z)
print(raw_error)


# In[4]:


x_hat, gamma = FB_recursion(TRANS, PI, z)
error = normalized_error_rate(x, x_hat, raw_error)
f.write('%.5f\n'%(error))
print('%.5f'%(error))

