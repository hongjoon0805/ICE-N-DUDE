
# coding: utf-8

# In[ ]:


# ICML_2019/DNA_CNN/Denoising/16map

from tools import *
import numpy as np
import sys
from keras import utils
import tensorflow as tf
import keras as K


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/16map

class ICE_Process:
    def __init__(self, n, k, nb_x_classes, nb_z_classes, x, z, param_name = 'test'):
        self.n = n
        self.k = k
        self.nb_x_classes = nb_x_classes
        self.nb_z_classes = nb_z_classes
        self.x = x
        self.z = z
        self.param_name = param_name
        self.C = make_batch(z, k, nb_z_classes)
    
    def Approximate_E_step(self, pred_prob): # approximate E-step & M-step
        n, k, nb_x_classes, nb_z_classes, z = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z
        
        """
        gamma[t][j] = p(x_t = j|Z_t,C_t;w)
        """
        
        # approximate E-step
        
        # reshape the output
        pred_prob = np.asarray(pred_prob)
        _,N,D,_ = pred_prob.shape
        pred_prob = np.resize(pred_prob, (nb_z_classes,N*D,nb_x_classes))[:,:n,:]
        gamma = pred_prob[z,np.arange(n)]
        
        return gamma
        
    def M_step(self, pred_prob):
        n, k, nb_x_classes, nb_z_classes, z = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z
        
        gamma = self.Approximate_E_step(pred_prob)
        
        # M-step
        PI = np.zeros((nb_x_classes, nb_z_classes))
        np.add.at(PI.T, self.z, gamma)
        PI /= (np.sum(gamma, axis = 0).reshape(nb_x_classes,1) + 1e-35)
        return PI
    
    def ICE(self, PI): # Iterative Channel Estimation Process
        n, k, nb_x_classes, nb_z_classes, z, param_name, C = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.param_name, self.C
        iteration = 3
        
        for t in range(iteration):
            
            # reset the L_new matrix
            L_new = L_NEW(PI)
            Y = make_pseudo_label(C, k, L_new)
            model = NDUDE_CNN_model(1000, nb_x_classes, nb_z_classes, k)
            
            # from second iteration, load previous weights and reset the learning rate.
            if t!=0:
                model.load_weights("weights/iteration/"+param_name+"_%d.hd5"%(t-1))
            
            # model training...
            hist = model.fit(C, [Y[:,:,0:4], Y[:,:,4:8], Y[:,:,8:12], Y[:,:,12:16]],
                             epochs=20, batch_size=5*4, verbose=1, 
                             validation_data=(C, [Y[:,:,0:4], Y[:,:,4:8], Y[:,:,8:12], Y[:,:,12:16]]))
            pred_prob = model.predict(C, batch_size = 500*4, verbose = 0)
            
            
            # estimate the channel
            PI = self.M_step(pred_prob)
            
            # save weights for next iteration
            model.save_weights("weights/iteration/"+param_name+"_%d.hd5"%(t))
            
            # save weights for denoising process
            if t == iteration-1:
                model.save_weights("weights/"+param_name+".hd5")
                save_PI(PI, param_name)
        return PI


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/16map

class Denoising_Process:
    def __init__(self, n, k, nb_x_classes, nb_z_classes, x, z, param_name = 'test'):
        self.n = n
        self.k = k
        self.nb_x_classes, self.nb_z_classes = nb_x_classes, nb_z_classes
        self.x = x
        self.z = z
        self.param_name = param_name
        self.raw_error = error_rate(x,z)
        self.C = make_batch(z, k, nb_z_classes)
    
    def Estimate(self, pred_prob): # Denoise sequence using softmax output
        n, k, nb_x_classes, nb_z_classes, z, x = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.x
        
        """
        pred_class[z][0] = 0
        pred_class[z][1] = 1
        pred_class[z][2] = 2
        pred_class[z][3] = 3
        """
        
        # reshape the output
        pred_prob = np.asarray(pred_prob)
        _,N,D,_ = pred_prob.shape
        pred_prob = np.resize(pred_prob, (nb_z_classes,N*D,nb_x_classes))[:,:n,:]
        gamma = pred_prob[z,np.arange(n)]
        
        x_hat = np.argmax(gamma, axis = 1)
        
        error = normalized_error_rate(x,x_hat, self.raw_error)
        return error, x_hat
    
    def N_DUDE(self, PI): # Denoising process
        n, k, nb_x_classes, nb_z_classes, z, param_name, C = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.param_name, self.C
        
        
        # fine-tuning the weights from ICE process
        L_new = L_NEW(PI)
        Y = make_pseudo_label(C, k, L_new)
        
        # model assign & train
        try:
            model = NDUDE_CNN_model(1000, nb_x_classes, nb_z_classes, k, lr = 0.0001)
            model.load_weights("weights/"+param_name+".hd5")
            hist = model.fit(C,[Y[:,:,0:4], Y[:,:,4:8], Y[:,:,8:12], Y[:,:,12:16]],
                             epochs=10, batch_size=5*4, verbose=1, 
                             validation_data=(C, [Y[:,:,0:4], Y[:,:,4:8], Y[:,:,8:12], Y[:,:,12:16]]))
            
            
        except:            
            model = NDUDE_CNN_model(1000, nb_x_classes, nb_z_classes, k)
            hist = model.fit(C,[Y[:,:,0:4], Y[:,:,4:8], Y[:,:,8:12], Y[:,:,12:16]],
                             epochs=20, batch_size=5*4, verbose=1, 
                             validation_data=(C, [Y[:,:,0:4], Y[:,:,4:8], Y[:,:,8:12], Y[:,:,12:16]]))
            
        pred_prob = model.predict(C, batch_size = 500*4, verbose = 0)
        
        return self.Estimate(pred_prob)


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/16map

class BW_1st_channel:
    def __init__(self, n, x, z, param_name = 'test'):
        self.n, self.x, self.z  = n, x, z
        self.param_name = param_name
        
    def Estimate(self, gamma):
        # use Bayes response
        x = self.x
        x_hat = np.argmax(gamma[1:], axis = 1)
        error = error_rate(x,x_hat)
        return error, x_hat
    
    def Baum_Welch(self, TRANS, PI):
        n, z, param_name = self.n, self.z, self.param_name
        
        a = np.copy(TRANS)
        b = np.copy(PI)
        
        T = z.shape[0]
        hid_states = a.shape[0]
        obs_states = b.shape[1]
        
        pi = np.ones(hid_states) / float(hid_states)
        gamma = None
        delta = None
        p = None
        #while True:
        for i in range(40):
            xi = np.zeros((T+1, 2, hid_states))
            gamma = np.zeros((T+1, hid_states))
            joint = np.zeros((T+1, hid_states, hid_states))

            for t in range(1,T+1): # 1~T
                eta = b[:, z[t-1]]
                if t==1:
                    xi[t][0] = pi
                else:
                    xi[t][0] = np.matmul(xi[t-1][1], a)
                xi[t][1] = (eta * xi[t][0]) / (np.sum(eta * xi[t][0]) + 1e-35)

            gamma[T] = xi[T][1]
            for t in reversed(range(1,T)):
                gamma[t] = xi[t][1] * np.matmul(gamma[t+1] / (xi[t+1][0]) + 1e-35, a.T)
                joint[t] = xi[t][1].reshape(hid_states,1) * (gamma[t+1] / (xi[t+1][0] + 1e-35)) * a


            a_before = a
            b_before = b
            pi = gamma[1]
            a = np.sum(joint[1:T], axis = 0) / (np.sum(gamma[1:T], axis = 0).reshape(hid_states,1) + 1e-35)

            b = b * 0
            np.add.at(b.T, z, gamma[1:])
            b /= (np.sum(gamma, axis = 0).reshape(hid_states,1) + 1e-35)


            if rel_error(a, a_before) < 1e-6 and rel_error(b, b_before) < 1e-6:
                break
        
        save_PI(b, param_name)
        return a, b, gamma
        


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/16map

class BW_2nd_channel:
    def __init__(self, n, nb_classes, x, z, param_name = 'test'):
        self.n = n
        self.nb_classes = nb_classes
        self.x = x
        self.z = z
        self.param_name = param_name
        self.raw_error = error_rate(x,z)
    def denoise(self, gamma):
        # use bayes response
        x_hat = np.argmax(gamma[1:], axis = 1)
        normalized_error = normalized_error_rate(x_hat%4, self.x%4, self.raw_error)
        return normalized_error, x_hat
    
    def Baum_Welch(self, TRANS, PI):
        n, nb_classes, z, param_name = self.n, self.nb_classes, self.z, self.param_name
        
        pi = np.ones(nb_classes*nb_classes) / float(nb_classes*nb_classes)
        
        a = np.copy(TRANS)
        b = np.copy(PI)
        
        T = z.shape[0]
        hid_states = a.shape[0]
        obs_states = b.shape[1]
        gamma = None
        delta = None
        p = None
        #while True:
        for i in range(40):
            xi = np.zeros((T+1, 2, hid_states))
            gamma = np.zeros((T+1, hid_states))
            joint = np.zeros((T+1, hid_states, hid_states))

            for t in range(1,T+1): # 1~T
                eta = b[:, z[t-1]]
                if t==1:
                    xi[t][0] = pi
                else:
                    xi[t][0] = np.matmul(xi[t-1][1], a)
                xi[t][1] = (eta * xi[t][0]) / (np.sum(eta * xi[t][0]) + 1e-35)

            gamma[T] = xi[T][1]
            for t in reversed(range(1,T)):
                gamma[t] = xi[t][1] * np.matmul(gamma[t+1] / (xi[t+1][0]) + 1e-35, a.T)
                joint[t] = xi[t][1].reshape(hid_states,1) * (gamma[t+1] / (xi[t+1][0] + 1e-35)) * a

            
            
            # marginalize the channel
            b_hat = np.zeros((obs_states, obs_states))
            gamma_hat = np.zeros((T+1, obs_states))
            for i in range(obs_states):
                arr = np.zeros(T+1)
                for j in range(obs_states):
                    arr += gamma[:,i+j*obs_states]
                gamma_hat[:,i] = arr

            np.add.at(b_hat.T, z, gamma_hat[1:])
            b_hat /= (np.sum(gamma_hat, axis = 0).reshape(obs_states,1) + 1e-35)
            
            
            a_before = a
            b_before = b
            pi = gamma[1]
            a = np.sum(joint[1:T], axis = 0) / (np.sum(gamma[1:T], axis = 0).reshape(hid_states,1) + 1e-35)

            b = b * 0
            np.add.at(b.T, z, gamma[1:])
            b /= (np.sum(gamma, axis = 0).reshape(hid_states,1) + 1e-35)


            if rel_error(a, a_before) < 1e-6 and rel_error(b, b_before) < 1e-6:
                break
        
        save_PI(b_hat, param_name)
        return a, b_hat, gamma_hat
        

