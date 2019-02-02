
# coding: utf-8

# In[1]:


import numpy as np
from numpy import *
from tools import *

import sys
import argparse


# In[2]:


try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--t", help="PI type", type=int)
    parser.add_argument("--x", help="the number of x classes", type=int)
    parser.add_argument("--k", help="window size k", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    nb_x_classes = args.x
    nb_z_classes = nb_x_classes
    k = args.k
    n = int(1e6)
    
except:
    result_name = "test"
    type_num = 1
    nb_x_classes = 4
    nb_z_classes = nb_x_classes
    n = int(1e6)
    k = 16
PI_type_arr = ['20%', '30%']
delta_arr = [0.2, 0.3]


# In[3]:


PI_type = PI_type_arr[type_num]
PI = sym_mat(nb_x_classes, delta_arr[type_num])
TRANS = sym_mat(nb_x_classes, 0.1)

print(PI)

f = open('results/' + '%d_'%nb_x_classes + PI_type + '_' + result_name,'a')

x, z = Hidden_Markov(n, TRANS, PI)

print(x[:20])
print(z[:20])

raw_error = error_rate(x,z)

print(raw_error)


# In[4]:


x_hat = dude(z, k, nb_x_classes, nb_z_classes, PI)
error = normalized_error_rate(x, x_hat, raw_error)
f.write('%d %.5f\n'%(k, error))
print('%d %.5f\n'%(k, error))

