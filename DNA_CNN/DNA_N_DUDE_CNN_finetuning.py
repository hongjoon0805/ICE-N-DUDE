
# coding: utf-8

# In[ ]:


from core import *
from tools import *
import numpy as np
import os
import tensorflow as tf
import keras as K

import sys
import argparse


# In[ ]:


try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--d", help="assumed delta", type=float)
    parser.add_argument("--k", help="window size k", type=int)
    parser.add_argument("--g", help="gpu number", type=int)
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    assumed_delta = args.d
    k = args.k
    gpu_num = args.g
    
except:
    result_name = "test"
    assumed_delta = 0.4
    k = 150
    gpu_num = 3


# In[ ]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.backend.set_session(session)


# In[ ]:


f = open('results/'+result_name,'a')

PI_true = array([ [ 0.8122,  0.0034,  0.0894,  0.0950],
                  [ 0.0096,  0.8237,  0.0808,  0.0859],
                  [ 0.1066,  0.0436,  0.7774,  0.0724],
                  [ 0.0704,  0.0690,  0.0889,  0.7717]])


x, z = load_DNA(PI_true)

n = len(x)

"""
nb_classes = 4
param_name = "ICE_%03d_%.2f"%(k, assumed_delta)
"""

print("k: %d assumed_delta: %.2f"%(k, assumed_delta))
nb_classes = 4
param_name = "ICE_%03d_%.2f"%(k, assumed_delta)

PI = load_PI(param_name)

DE = Denoising_Process(n, k, nb_classes, x, z, param_name)

# denoise the image using estimated PI & weights from ICE-N-DUDE
normalized_error, x_hat = DE.N_DUDE(PI)
f.write("%d %.2f %.5f\n"%(k, assumed_delta, normalized_error))
f.flush()
print(normalized_error)

f.close()

