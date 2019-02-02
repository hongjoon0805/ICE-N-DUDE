
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


# ICML_2019/DNA_CNN/Denoising

class ICE_Process:
    def __init__(self, n, k, nb_x_classes, nb_z_classes, x, z, param_name = 'test'):
        self.n, self.k, self.x, self.z, self.nb_x_classes, self.nb_z_classes = n, k, x, z, nb_x_classes, nb_z_classes
        self.param_name = param_name
        self.raw_error = error_rate(x,z)
        self.C = make_batch(z,k)
    
    def Approximate_E_step(self, pred_prob): # approximate E-step & M-step
        n, k, z, nb_x_classes, nb_z_classes = self.n, self.k, self.z, self.nb_x_classes, self.nb_z_classes
        
        """
        gamma[t][j] = p(x_t = j|Z_t,C_t;w)
        """
        
        # approximate E-step
        
        gamma = np.zeros((n, nb_x_classes))
        for i in range(nb_x_classes):
            gamma[:,i] = pred_prob[:,i+1]
        gamma[np.arange(n), z] += pred_prob[np.arange(n), 0]
        return gamma
        
    def M_step(self, pred_prob):
        n, k, z, nb_x_classes, nb_z_classes = self.n, self.k, self.z, self.nb_x_classes, self.nb_z_classes
        
        gamma = self.Approximate_E_step(pred_prob)
        
        # M-step
        PI = np.zeros((nb_x_classes, nb_z_classes))
        np.add.at(PI.T, z, gamma)
        PI /= (np.sum(gamma, axis = 0).reshape(nb_x_classes,1) + 1e-35)
        return PI
    
    def ICE(self, PI_true): # Iterative Channel Estimation Process
        n, k, nb_x_classes, nb_z_classes, z, param_name, C = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.param_name, self.C
        iteration = 10
        
        train_loss, val_loss = [], []
        
        for t in range(iteration):
            # reset the L_new matrix
            L_new_true = L_NEW(PI_true, nb_x_classes, nb_z_classes)
            Y_true = make_pseudo_label(C, k, L_new_true)
            
            model = NDUDE_CNN_model_5map(1000, nb_x_classes, nb_z_classes, k)
            
            # from second iteration, load previous weights and reset the learning rate.
            if t!=0:
                model.load_weights("weights/iteration/"+param_name+"_%d.hd5"%(t-1))
            
            # model training...
            hist = model.fit(C,Y_true,epochs=20, batch_size=2*4, verbose=1, validation_data=(C, Y_true))
            pred_prob = model.predict(C, batch_size = 20*4, verbose = 0)
            
            train_loss.append(hist.history['loss'][-1])
            val_loss.append(hist.history['val_loss'][-1])
            
            
            # resize the output
            N,D,_ = pred_prob.shape
            pred_prob = np.resize(pred_prob, (N*D,5))[:n]
            
            # estimate the channel
            PI = self.M_step(pred_prob)
            
            # save weights for next iteration
            model.save_weights("weights/iteration/"+param_name+"_%d.hd5"%(t))
            save_PI(PI, param_name + '_%d'%(t-1))
            
            # save weights for denoising process
            if t == iteration-1:
                model.save_weights("weights/"+param_name+".hd5")
                save_PI(PI, param_name)
        return PI, train_loss, val_loss


# In[ ]:


PI_true = array([ [ 0.8122,  0.0034,  0.0894,  0.0950],
                  [ 0.0096,  0.8237,  0.0808,  0.0859],
                  [ 0.1066,  0.0436,  0.7774,  0.0724],
                  [ 0.0704,  0.0690,  0.0889,  0.7717]])
x, z = load_DNA(PI_true)

try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--k", help="window size k", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    k = args.k
    n = len(x)
    
except:
    result_name = "test"
    k = 150
    n = int(1e4)
    n = len(x)

nb_x_classes, nb_z_classes = 4, 4
param_name = 'NDUDE'
x, z = x[:n], z[:n]


# In[ ]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2, 3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.backend.set_session(session)


# In[ ]:


f = open('results/'+result_name,'a')
print("k: %d "%(k))

ICE_N_DUDE = ICE_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
Estimated_PI, train_loss, val_loss = ICE_N_DUDE.ICE(PI_true)
print(Estimated_PI)

for i in range(10):
    f.write('train %d %.5f\n'%(i+1,train_loss[i]))

for i in range(10):
    f.write('validation %d %.5f\n'%(i+1,val_loss[i]))

