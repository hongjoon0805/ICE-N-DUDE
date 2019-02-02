
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


try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--x", help="the number of x classes", type=int)
    parser.add_argument("--d", help="assumed delta", type=float)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = 1
    nb_x_classes = args.x
    nb_z_classes = nb_x_classes
    assumed_delta = args.d
    k = 16
    n = int(1e6)
    
except:
    result_name = "test"
    type_num = 1
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    assumed_delta = 0.3
    n = int(1e6)
    k = 16
PI_type_arr = ['20%', '30%']
delta_arr = [0.2, 0.3]


# In[ ]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.backend.set_session(session)


# In[ ]:


PI_type = PI_type_arr[type_num]
PI = sym_mat(nb_x_classes, delta_arr[type_num])
TRANS = sym_mat(nb_x_classes, 0.1)
assumed_PI = sym_mat(nb_x_classes, assumed_delta)

print(PI)

f = open('results/' + '%d_'%nb_x_classes + PI_type + '_' + result_name,'a')

param_name = "NDUDE"


x, z = Hidden_Markov(n, TRANS, PI)

print(x[:20])
print(z[:20])

print(error_rate(x,z))


# In[ ]:


# State Estimation Process
SE = State_Estimation_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
error, x_hat = SE.N_DUDE(assumed_PI)
f.write("%.2f %.5f\n"%(assumed_delta, error))
print('%.2f %.5f'%(assumed_delta, error))

