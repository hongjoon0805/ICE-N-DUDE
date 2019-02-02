
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


# In[4]:


# ICML_2019/image

class State_Estimation_Process:
    def __init__(self, n, k, nb_x_classes, nb_z_classes, x, z, param_name = 'test'):
        self.n, self.k, self.x, self.z, self.nb_x_classes, self.nb_z_classes = n, k, x, z, nb_x_classes, nb_z_classes
        self.param_name = param_name
        self.raw_error = error_rate(x,z)
        self.C = make_context(z, k, nb_z_classes, n)
        
        self.train_batch_size = 100 + 200 * (nb_x_classes - 2)
        self.test_batch_size = 3000
        self.epochs = nb_z_classes * 5
    
    def denoise(self, pred_prob): # Estimate latent variables using softmax output
        n, k, x, z = self.n, self.k, self.x, self.z
        
        """
        pred_class[0] = Say What You See(s[0]=z[i]) = -1
        pred_class[i+1] = Always Say i(s[i+1]=i) = i
        """
        
        # s(z) = z
        pred_class = np.argmax(pred_prob, axis = -1) - 1
        
        # mask Say What You see
        mask = pred_class == -1
        
        # mask-> Say What You see || others-> 0,1,2,3
        x_hat = z[k:n-k] * mask + (mask^1)*pred_class
        x_hat = np.hstack((z[:k], x_hat, z[n-k:n]))
        
        error = normalized_error_rate(x,x_hat,self.raw_error)
        return error, x_hat
    
    def N_DUDE(self, PI): # Denoising process
        n, k, nb_x_classes, nb_z_classes, z, param_name, C = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.param_name, self.C
        epochs, train_batch_size, test_batch_size = self.epochs, self.train_batch_size, self.test_batch_size
        iteration = 3
            
        # fine-tuning the weights from ICE process
        L_new = L_NEW(PI, nb_x_classes, nb_z_classes)
        Y = make_pseudo_label(z, k, L_new, nb_z_classes, n)
        model = ICE_N_DUDE_model(nb_x_classes, nb_z_classes, k, lr = 0.0001)
        model.load_weights("weights/"+param_name+".hd5")
        # model training...
        hist = model.fit(C, Y, epochs=epochs // 2, batch_size=train_batch_size*4, verbose=1, validation_data=(C, Y))
        
        model.load_weights("weights/"+param_name+".hd5")
        pred_prob = model.predict(C, batch_size = test_batch_size*4, verbose = 0)
        return self.denoise(pred_prob)


# In[ ]:


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
    img_num = 0
    k = 50
PI_type_arr = ['20%', '30%', '10%']


# In[ ]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.backend.set_session(session)


# In[ ]:


PI_type = PI_type_arr[type_num]
PI = load_channel('true', nb_x_classes, 1, type_num)

print(PI)

f = open('results/' + '%d_'%nb_x_classes + PI_type + '_' + result_name,'a')

x,z = load_img(PI, [img_arr[img_num]])

param_name = "NDUDE_%d"%(type_num)

n = len(x)

print(x[:20])
print(z[:20])

print(n)

print(error_rate(x,z))


# In[ ]:


# State Estimation Process
SE = State_Estimation_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
error, x_hat = SE.N_DUDE(PI)
f.write("%d %.5f\n"%(img_num, error))
print('%d %.5f'%(img_num, error))

