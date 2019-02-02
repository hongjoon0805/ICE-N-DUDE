
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
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    assumed_delta = args.d
except: # default setting
    result_name = "test"
    assumed_delta = 0.40


# In[ ]:


f = open('results/'+result_name,'a')

PI_true = array([ [ 0.8122,  0.0034,  0.0894,  0.0950],
                  [ 0.0096,  0.8237,  0.0808,  0.0859],
                  [ 0.1066,  0.0436,  0.7774,  0.0724],
                  [ 0.0704,  0.0690,  0.0889,  0.7717]])


x, z = load_DNA(PI_true)

n = len(x)

alpha1 = 0.05
alpha2 = 0.10
alpha3 = 0.20
alpha4 = 0.30

assumed_TRANS = np.array([[1-alpha1,alpha1/3,alpha1/3,alpha1/3,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,alpha1/3,1-alpha1,alpha1/3,alpha1/3,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,alpha1/3,alpha1/3,1-alpha1,alpha1/3,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,alpha1/3,alpha1/3,alpha1/3,1-alpha1],
                          [1-alpha2,alpha2/3,alpha2/3,alpha2/3,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,alpha2/3,1-alpha2,alpha2/3,alpha2/3,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,alpha2/3,alpha2/3,1-alpha2,alpha2/3,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,alpha2/3,alpha2/3,alpha2/3,1-alpha2],
                          [1-alpha3,alpha3/3,alpha3/3,alpha3/3,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,alpha3/3,1-alpha3,alpha3/3,alpha3/3,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,alpha3/3,alpha3/3,1-alpha3,alpha3/3,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,alpha3/3,alpha3/3,alpha3/3,1-alpha3],
                          [1-alpha4,alpha4/3,alpha4/3,alpha4/3,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,alpha4/3,1-alpha4,alpha4/3,alpha4/3,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,alpha4/3,alpha4/3,1-alpha4,alpha4/3,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,alpha4/3,alpha4/3,alpha4/3,1-alpha4]])

assumed_PI = np.array([[1-assumed_delta,assumed_delta/3,assumed_delta/3,assumed_delta/3],
                       [assumed_delta/3,1-assumed_delta,assumed_delta/3,assumed_delta/3],
                       [assumed_delta/3,assumed_delta/3,1-assumed_delta,assumed_delta/3],
                       [assumed_delta/3,assumed_delta/3,assumed_delta/3,1-assumed_delta],
                       [1-assumed_delta,assumed_delta/3,assumed_delta/3,assumed_delta/3],
                       [assumed_delta/3,1-assumed_delta,assumed_delta/3,assumed_delta/3],
                       [assumed_delta/3,assumed_delta/3,1-assumed_delta,assumed_delta/3],
                       [assumed_delta/3,assumed_delta/3,assumed_delta/3,1-assumed_delta],
                       [1-assumed_delta,assumed_delta/3,assumed_delta/3,assumed_delta/3],
                       [assumed_delta/3,1-assumed_delta,assumed_delta/3,assumed_delta/3],
                       [assumed_delta/3,assumed_delta/3,1-assumed_delta,assumed_delta/3],
                       [assumed_delta/3,assumed_delta/3,assumed_delta/3,1-assumed_delta],
                       [1-assumed_delta,assumed_delta/3,assumed_delta/3,assumed_delta/3],
                       [assumed_delta/3,1-assumed_delta,assumed_delta/3,assumed_delta/3],
                       [assumed_delta/3,assumed_delta/3,1-assumed_delta,assumed_delta/3],
                       [assumed_delta/3,assumed_delta/3,assumed_delta/3,1-assumed_delta]])


"""
nb_classes = 4
param_name = "BW_2nd_%.2f"%assumed_delta
"""

nb_classes = 4
param_name = "BW_2nd_%.2f"%assumed_delta

BW_2nd = BW_2nd_channel(n, nb_classes, x, z, param_name)

# estimate the posterior
_, _, gamma = BW_2nd.Baum_Welch(assumed_TRANS, assumed_PI)

# denoise the image
normalized_error, x_hat = BW_2nd.denoise(gamma)
print(normalized_error)
f.write("%.2f %.5f\n"%(assumed_delta, normalized_error))
f.flush()

f.close()

