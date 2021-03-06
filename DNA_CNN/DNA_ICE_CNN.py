
# coding: utf-8

# In[1]:


from core import *
from tools import *
import numpy as np
import os
import tensorflow as tf
import keras as K

import sys
import argparse


# In[2]:


PI_true = array([ [ 0.8122,  0.0034,  0.0894,  0.0950],
                  [ 0.0096,  0.8237,  0.0808,  0.0859],
                  [ 0.1066,  0.0436,  0.7774,  0.0724],
                  [ 0.0704,  0.0690,  0.0889,  0.7717]])
x, z = load_DNA(PI_true)

try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--d", help="Assumed delta", type=float)
    parser.add_argument("--k", help="window size k", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    k = args.k
    assumed_delta = args.d
    n = len(x)
    param_name = "ICE_%03d_%.2f"%(k, assumed_delta)
        
except:
    result_name = "test"
    k = 150
    assumed_delta = 0.40
    n = int(1e4)
    param_name = "test_%03d_%.2f"%(k, assumed_delta)
    
nb_x_classes, nb_z_classes = 4, 4
assumed_PI = sym_mat(4, assumed_delta)
x, z = x[:n], z[:n]


# In[3]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.backend.set_session(session)


# In[4]:


f = open('results/'+result_name,'a')
print("k: %d assumed_delta: %.2f"%(k,assumed_delta))

ICE_N_DUDE = ICE_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
Estimated_PI = ICE_N_DUDE.ICE(assumed_PI)
print(Estimated_PI)

DE = Denoising_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
normalized_error, x_hat = DE.N_DUDE(Estimated_PI)
f.write("%d %.2f %.5f\n"%(k, assumed_delta, normalized_error))
print(normalized_error)

