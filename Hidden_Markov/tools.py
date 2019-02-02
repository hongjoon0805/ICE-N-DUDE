
# coding: utf-8

# In[1]:


# ICML_2019/Hidden_Markov/Denoising

import numpy as np
from numpy import *

import tensorflow as tf
import keras as K
from keras import layers, optimizers, models, utils
from keras.layers import Input, Dense, Activation, Add
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from PIL import Image
import scipy.io as sio
import h5py

from keras.engine.topology import Layer


# In[2]:


# ICML_2019/Hidden_Markov/Denoising

def error_rate(a,b):
    error = absolute(a-b) > 0
    return np.mean(error)

def normalized_error_rate(a,b,raw_error):
    error = absolute(a-b) > 0
    return np.mean(error) / raw_error

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# In[4]:


# ICML_2019/Hidden_Markov/Denoising

def L_NEW(PI, nb_x_classes, nb_z_classes):
    PI_INV = linalg.inv(PI)
    RHO = np.zeros((nb_x_classes, nb_x_classes+1))
    LAMBDA = np.ones((nb_x_classes, nb_x_classes)) - np.eye(nb_x_classes)

    MAP = np.ones((nb_x_classes, nb_x_classes+1), dtype = int)

    for x in range(nb_x_classes):
        for s in range(nb_x_classes+1):
            MAP[x][s] = s - 1
            MAP[x][0] = x

    for x in range(nb_x_classes):
        for s in range(nb_x_classes+1):
            for z in range(nb_z_classes):
                RHO[x][s] += PI[x][z] * LAMBDA[x][MAP[z][s]]

    L = np.matmul(PI_INV, RHO)
    L_new = -L + amax(L)
    return L_new


# In[5]:


# ICML_2019/Hidden_Markov/Denoising

def load_PI(name):
    hdf5_path = 'PI/'+name+'.hdf5'
    hdf5_file = h5py.File(hdf5_path, mode='r')
    PI = hdf5_file[name][...]
    hdf5_file.close()
    return PI

def save_PI(PI, name):
    hdf5_path = 'PI/'+name+'.hdf5'
    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset(name, PI.shape, np.float32, data = PI)
    hdf5_file.close()

def load_channel(true_or_assumed, nb_x_classes, order, PI_type_num):
    PI_dict = sio.loadmat('PARAM/' + true_or_assumed + '_PI.mat')
    PI = PI_dict['%d_%d_%d'%(nb_x_classes, order, PI_type_num)]
    return PI

def load_TRANS(true_or_assumed, nb_x_classes, order):
    TRANS_dict = sio.loadmat('PARAM/' + true_or_assumed + '_TRANS.mat')
    TRANS = TRANS_dict['%d_%d'%(nb_x_classes, order)]
    return TRANS


# In[7]:


# ICML_2019/Hidden_Markov/Denoising

def sym_mat(states, prob):
    x = ones((states,states)) * (prob/(states-1))
    for i in range(states):
        x[i][i] = 1 - (states-1)*x[i][i]
    return x

def convert_sequence(x, order, nb_x_classes):
    if order == 1:
        return x
    x_temp = np.copy(x)
    x_temp = np.hstack((np.zeros(order-1), x_temp))
    n = len(x)
    mask = np.ones((order))
    x_convert = np.zeros(n, dtype = int)
    for i in range(order):
        mask[i] = nb_x_classes ** (order-i-1)    
    
    for i in range(n):
        x_convert[i] = np.dot(mask, x_temp[i:i+order])
    
    return x_convert

def make_context(z, k, nb_z_classes, n):
    Z = utils.np_utils.to_categorical(z,nb_z_classes)
    c_length=2*k
    C=zeros((n-2*k, 2*k*nb_z_classes))

    for i in range(k,n-k):
        c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_z_classes)
        C[i-k,]=c_i
        
    return C

def make_pseudo_label(z, k, L_new, nb_z_classes, n):
    Z = utils.np_utils.to_categorical(z, nb_z_classes)
    Y = dot(Z[k:n-k],L_new)
    return Y


# In[9]:


# ICML_2019/Hidden_Markov/Denoising

def Hidden_Markov(n, a, b):
    x, z = np.zeros(n, dtype = np.int), np.zeros(n, dtype = np.int)
    hid_states, obs_states = a.shape[0], b.shape[1]
    a_sum = np.copy(a)
    b_sum = np.copy(b)

    for i in range(1,hid_states):
        a_sum.T[i] += a_sum.T[i-1]
    for i in range(1,obs_states):
        b_sum.T[i] += b_sum.T[i-1]
    
    
    prob = np.random.random()
    x[0] = int(prob/(1 / float(hid_states)))
    prob = np.random.random()
    z[0] = np.argmax(b_sum[x[0]] > prob)
    
    for i in range(1,n):
        prob = np.random.random()
        x[i] = np.argmax(a_sum[x[i-1]] > prob)
        prob = np.random.random()
        z[i] = np.argmax(b_sum[x[i]] > prob)
    return x, z


# In[10]:


# ICML_2019/Hidden_Markov/Denoising

def dude(z, k, nb_x_classes, nb_z_classes, PI):
    n=len(z)
    x_hat=np.zeros(n,dtype=np.int)
    m={}
    PI_INV = linalg.inv(PI)
    LAMBDA = np.ones((nb_x_classes, nb_x_classes)) - np.eye(nb_x_classes)
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if context_str not in m:
            m[context_str]=np.zeros(nb_z_classes,dtype=np.int)
            m[context_str][z[i]]=1
        else:
            m[context_str][z[i]]+=1
    x_hat[:k] = z[:k]
    x_hat[n-k:n] = z[n-k:n]
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        m_vector = m[context_str]
        EXP = np.dot(PI_INV, LAMBDA * (PI[:,z[i]].reshape((nb_x_classes,1))))
        score = np.dot(m_vector, EXP)
        x_hat[i] = np.argmin(score)
    
    return x_hat


# In[11]:


# ICML_2019/Hidden_Markov/Denoising

def FB_recursion(a, b, z, denoise = True):
    
    T = z.shape[0]
    hid_states = a.shape[0]
    obs_states = b.shape[1]
    xi = np.zeros((T+1, 2, hid_states))
    gamma = np.zeros((T+1, hid_states))
    
    pi = np.ones(hid_states) / float(hid_states)
    
    for t in range(1,T+1): # 1~T
        eta = b[:, z[t-1]]
        if t==1:
            xi[t][0] = pi
        else:
            xi[t][0] = np.matmul(xi[t-1][1], a)    
        xi[t][1] = (eta * xi[t][0]) / (np.sum(eta * xi[t][0]) + 1e-35)
        
    gamma[T] = xi[T][1]
    for t in reversed(range(1,T)):
        gamma[t] = xi[t][1] * np.matmul(gamma[t+1] / (xi[t+1][0] + 1e-35) , a.T)
    
    if denoise == False:
        return gamma
    #denoise the sequence
    x_hat = np.argmax(gamma[1:], axis = 1)

    return x_hat, gamma


# In[13]:


# ICML_2019/Hidden_Markov/Denoising

def ICE_N_DUDE_model(nb_x_classes, nb_z_classes, k, lr = 0.001):
    unitN = 20 * nb_z_classes
    with tf.device('/cpu:0'):
        
        inputs = Input(shape=(2*k*nb_z_classes,))
        layer = layers.Dense(unitN)(inputs)
        layer = layers.Activation('relu')(layer)
        layer = layers.Dense(unitN)(layer)
        layer = layers.Activation('relu')(layer)
        layer = layers.Dense(unitN)(layer)
        layer = layers.Activation('relu')(layer)
        layer = layers.Dense(nb_x_classes+1)(layer)
        output = layers.Activation('softmax')(layer)
        model = models.Model(inputs = inputs, outputs = output)
    
    adam = optimizers.Adam(lr=lr)
    multi_model = multi_gpu_model(model, gpus=4)
    multi_model.compile(loss='poisson', optimizer=adam)
    return multi_model

