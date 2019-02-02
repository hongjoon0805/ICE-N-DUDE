
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


# ICML_2019/Hidden_Markov/Denoising

class ICE_Process:
    def __init__(self, n, k, nb_x_classes, nb_z_classes, x, z, param_name = 'test'):
        self.n, self.k, self.x, self.z, self.nb_x_classes, self.nb_z_classes = n, k, x, z, nb_x_classes, nb_z_classes
        self.param_name = param_name
        self.raw_error = error_rate(x,z)
        self.C = make_context(z, k, nb_z_classes, n)
        
        self.train_batch_size = 100 + 200 * (nb_x_classes - 2)
        self.test_batch_size = 3000
        self.epochs = nb_z_classes * 5
    
    def Approximate_E_step(self, pred_prob): # approximate E-step & M-step
        n, k, nb_x_classes, nb_z_classes, z = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z
        
        """
        gamma[t][j] = p(x_t = j|Z_t,C_t;w)
        """
        
        # approximate E-step
        gamma = pred_prob[:,1:]
        """
        for i in range(nb_x_classes):
            gamma[:,i] = pred_prob[:,i+1]
        """
        gamma[np.arange(n-2*k), z[k:n-k]] += pred_prob[np.arange(n-2*k), 0]
        
        return gamma
    
    def M_step(self, pred_prob):
        n, k, nb_x_classes, nb_z_classes, z = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z
        
        gamma = self.Approximate_E_step(pred_prob)
        
        # M-step
        PI = np.zeros((nb_x_classes, nb_z_classes))
        np.add.at(PI.T, self.z[k:n-k], gamma)
        PI /= (np.sum(gamma, axis = 0).reshape(nb_x_classes,1) + 1e-35)
        return PI
    
    def ICE(self, PI_true): # Iterative Channel Estimation Process
        n, k, nb_x_classes, nb_z_classes, z, param_name, C = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.param_name, self.C
        epochs, train_batch_size, test_batch_size = self.epochs, self.train_batch_size, self.test_batch_size
        iteration = 10
        
        train_loss, val_loss = [], []
        
        for t in range(iteration):
            
            # reset the L_new matrix            
            L_new_true = L_NEW(PI_true, nb_x_classes, nb_z_classes)
            Y_true = make_pseudo_label(z, k, L_new_true, nb_z_classes, n)
                
            model = ICE_N_DUDE_model(nb_x_classes, nb_z_classes, k, lr = 0.001)
            
            # from second iteration, load previous weights and reset the learning rate.
            if t!=0:
                model.load_weights("weights/iteration/"+param_name+"_%d.hd5"%(t-1))
            
            # model training...
            hist = model.fit(C, Y_true, epochs=epochs, batch_size=train_batch_size*4, verbose=1, validation_data=(C, Y_true))
            pred_prob = model.predict(C, batch_size = test_batch_size*4, verbose = 0)
            
            train_loss.append(hist.history['loss'][-1])
            val_loss.append(hist.history['val_loss'][-1])
            
            PI = self.M_step(pred_prob)
            
            # save weights for next iteration
            model.save_weights("weights/iteration/"+param_name+"_%d.hd5"%(t))
            
        # save weights for denoising process
        model.save_weights("weights/"+param_name+".hd5")
        save_PI(PI, param_name)
        return PI, train_loss, val_loss


# In[ ]:


img_arr = ['barbara_512.png', 'boat_512.png', 'cman_256.png', 'couple_512.png', 'Einstein_256.jpeg', 'fruit_256.bmp', 'lena_512.jpg', 'pepers_256.png']
try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--t", help="PI type", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    k = 50
    
except:
    result_name = "test"
    type_num = 1
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
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

f = open('results/' + '%d_%s'%(nb_x_classes, PI_type) + '_' + result_name,'a')

param_name = "NDUDE_%d_%d"%(nb_x_classes, type_num)

x, z = load_img(PI, img_arr)

n = len(x)

print(x[:20])
print(z[:20])

print(n)

print(error_rate(x,z))


# In[ ]:


# Parameter Estimation Process
ICE_N_DUDE = ICE_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
Estimated_PI, train_loss, val_loss = ICE_N_DUDE.ICE(PI)

for i in range(10):
    f.write('train %d %.5f\n'%(i+1,train_loss[i]))

for i in range(10):
    f.write('validation %d %.5f\n'%(i+1,val_loss[i]))

print(Estimated_PI)

