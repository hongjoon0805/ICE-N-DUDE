
# coding: utf-8

# In[ ]:


# ICML_2019/DNA_CNN/Denoising

from tools import *
import numpy as np
import sys
from keras import utils
import tensorflow as tf
import keras as K


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
    
    def ICE(self, PI): # Iterative Channel Estimation Process
        n, k, nb_x_classes, nb_z_classes, z, param_name, C = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.param_name, self.C
        iteration = 3
        
        for t in range(iteration):
            # reset the L_new matrix
            L_new = L_NEW(PI, nb_x_classes, nb_z_classes)
            Y = make_pseudo_label(C, k, L_new)
            model = NDUDE_CNN_model_5map(1000, nb_x_classes, nb_z_classes, k)
            
            # from second iteration, load previous weights and reset the learning rate.
            if t!=0:
                model.load_weights("weights/iteration/"+param_name+"_%d.hd5"%(t-1))
            
            # model training...
            hist = model.fit(C,Y,epochs=20, batch_size=2*4, verbose=1, validation_data=(C, Y))
            pred_prob = model.predict(C, batch_size = 20*4, verbose = 0)
            
            # resize the output
            N,D,_ = pred_prob.shape
            pred_prob = np.resize(pred_prob, (N*D,nb_x_classes+1))[:n]
            
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


# ICML_2019/DNA_CNN/Denoising

class Denoising_Process:
    def __init__(self, n, k, nb_x_classes, nb_z_classes, x, z, param_name = 'test'):
        self.n, self.k, self.x, self.z, self.nb_x_classes, self.nb_z_classes = n, k, x, z, nb_x_classes, nb_z_classes
        self.param_name = param_name
        self.raw_error = error_rate(x,z)
        self.C = make_batch(z,k)
    
    def denoise(self, pred_prob): # Denoise sequence using softmax output
        n, k, x, z = self.n, self.k, self.x, self.z
        
        """
        pred_class[0] = Say What You See(s[0]=z[i]) = -1
        pred_class[1] = Always Say 0(s[1]=0) = 0
        pred_class[2] = Always Say 1(s[2]=1) = 1
        pred_class[3] = Always Say 1(s[2]=1) = 2
        pred_class[4] = Always Say 1(s[2]=1) = 3
        """
        
        # s(z) = z
        pred_class = np.argmax(pred_prob, axis = -1) - 1
        
        # mask Say What You see
        mask = pred_class == -1
        
        # mask-> Say What You see || others-> 0,1,2,3
        x_hat = z * mask + (mask^1)*pred_class

        error = normalized_error_rate(x,x_hat,self.raw_error)
        return error, x_hat
    
    def N_DUDE(self, PI): # Denoising process
        n, k, nb_x_classes, nb_z_classes, z, param_name, C = self.n, self.k, self.nb_x_classes, self.nb_z_classes, self.z, self.param_name, self.C
        
        
        # fine-tuning the weights from ICE process
        L_new = L_NEW(PI, nb_x_classes, nb_z_classes)
        Y = make_pseudo_label(C, k, L_new)
        
        # model assign & train
        if param_name == 'NDUDE':
            model = NDUDE_CNN_model_5map(1000, nb_x_classes, nb_z_classes, k)
            hist = model.fit(C,Y,epochs=20, batch_size=2*4, verbose=1, validation_data=(C, Y))
        
        else:
            model = NDUDE_CNN_model_5map(1000, nb_x_classes, nb_z_classes, k, lr = 0.0001)
            model.load_weights("weights/"+param_name+".hd5")
            hist = model.fit(C,Y,epochs=10, batch_size=2*4, verbose=1, validation_data=(C, Y))
            
        pred_prob = model.predict(C, batch_size = 20*4, verbose = 0)
        
        # reshape the output
        N,D,_ = pred_prob.shape
        pred_prob = np.resize(pred_prob, (N*D,nb_x_classes+1))[:n]
        
        return self.denoise(pred_prob)


# In[ ]:


# ICML_2019/DNA_CNN/Denoising

class BW_1st_channel:
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
        normalized_error = normalized_error_rate(x_hat, self.x, self.raw_error)
        return normalized_error, x_hat
    
    def Baum_Welch(self, TRANS, PI):
        n, nb_classes, z, param_name = self.n, self.nb_classes, self.z, self.param_name
        
        pi = np.ones(nb_classes) / float(nb_classes)
        
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


# ICML_2019/DNA_CNN/Denoising

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
        

