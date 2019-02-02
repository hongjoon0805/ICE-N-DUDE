
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
    
    def ICE(self, PI, PI_true): # Iterative Channel Estimation Process
        n, k, nb_x_classes, nb_z_classes, z, param_name, C = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.param_name, self.C
        epochs, train_batch_size, test_batch_size = self.epochs, self.train_batch_size, self.test_batch_size
        iteration = 10
        
        train_loss, val_loss = [], []
        
        for t in range(iteration):
            
            # reset the L_new matrix
            L_new = L_NEW(PI, nb_x_classes, nb_z_classes)
            Y = make_pseudo_label(z, k, L_new, nb_z_classes, n)
            
            L_new_true = L_NEW(PI_true, nb_x_classes, nb_z_classes)
            Y_true = make_pseudo_label(z, k, L_new_true, nb_z_classes, n)
                
            model = ICE_N_DUDE_model(nb_x_classes, nb_z_classes, k, lr = 0.001)
            
            # from second iteration, load previous weights and reset the learning rate.
            if t!=0:
                model.load_weights("weights/iteration/"+param_name+"_%d.hd5"%(t-1))
            
            # model training...
            hist = model.fit(C, Y, epochs=epochs, batch_size=train_batch_size*4, verbose=1, validation_data=(C, Y_true))
            pred_prob = model.predict(C, batch_size = test_batch_size*4, verbose = 0)
            
            train_loss.append(hist.history['loss'][-1])
            val_loss.append(hist.history['val_loss'][-1])
            
            PI = self.M_step(pred_prob)
            
            # save weights for next iteration
            model.save_weights("weights/iteration/"+param_name+"_%d.hd5"%(t))
            save_PI(PI, param_name + '_%d'%(t-1))
            
        # save weights for denoising process
        model.save_weights("weights/"+param_name+".hd5")
        save_PI(PI, param_name)
        return PI, train_loss, val_loss


# In[ ]:


img_arr = ['2012_000003.jpg', '2012_000004.jpg', '2012_000007.jpg', '2012_000010.jpg', '2012_000014.jpg', '2012_000015.jpg', '2012_000016.jpg', '2012_000019.jpg', '2012_000025.jpg', '2012_000027.jpg', '2012_000028.jpg', '2012_000029.jpg', '2012_000030.jpg', '2012_000031.jpg', '2012_000032.jpg', '2012_000035.jpg', '2012_000036.jpg', '2012_000040.jpg', '2012_000042.jpg', '2012_000044.jpg', '2012_000045.jpg', '2012_000049.jpg', '2012_000050.jpg', '2012_000051.jpg', '2012_000055.jpg', '2012_000056.jpg', '2012_000058.jpg', '2012_000059.jpg', '2012_000060.jpg', '2012_000065.jpg', '2012_000067.jpg', '2012_000069.jpg', '2012_000070.jpg', '2012_000071.jpg', '2012_000072.jpg', '2012_000074.jpg', '2012_000078.jpg', '2012_000083.jpg', '2012_000084.jpg', '2012_000085.jpg', '2012_000086.jpg', '2012_000087.jpg', '2012_000089.jpg', '2012_000100.jpg', '2012_000102.jpg', '2012_000104.jpg', '2012_000105.jpg', '2012_000106.jpg', '2012_000108.jpg', '2012_000113.jpg']

try:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--t", help="PI type", type=int)
    
    args = parser.parse_args()
    
    result_name = sys.argv[0]
    type_num = args.t
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    assumed_delta = 0.1
    k = 50
    
except:
    result_name = "test"
    type_num = 1
    nb_x_classes = 2
    nb_z_classes = nb_x_classes
    assumed_delta = 0.1
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
assumed_PI = load_channel('assumed', nb_x_classes, 1, type_num)

print(PI)
print(assumed_PI)

f = open('results/' + '%d_%s'%(nb_x_classes, PI_type) + '_' + result_name,'a')

param_name = "ICE_%d_%d"%(nb_x_classes, type_num)


x, z = load_img(PI, img_arr[:10])

n = len(x)

print(x[:20])
print(z[:20])

print(n)

print(error_rate(x,z))


# In[ ]:


# Parameter Estimation Process
ICE_N_DUDE = ICE_Process(n, k, nb_x_classes, nb_z_classes, x, z, param_name = param_name)
Estimated_PI, train_loss, val_loss = ICE_N_DUDE.ICE(assumed_PI, PI)

for i in range(10):
    f.write('train %d %.5f\n'%(i+1,train_loss[i]))

for i in range(10):
    f.write('validation %d %.5f\n'%(i+1,val_loss[i]))

print(Estimated_PI)

