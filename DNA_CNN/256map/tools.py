
# coding: utf-8

# In[1]:


# ICML_2019/DNA_CNN/Denoising/256map

import numpy as np
from numpy import *
from layers import masked_CNN

import keras
from keras.models import Model
from keras import layers, optimizers, models, utils
from keras.layers import Input, Activation, Add, Conv1D
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from sklearn.preprocessing import LabelBinarizer

import h5py


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/256map

def error_rate(a,b):
    error = absolute(a-b) > 0
    return np.mean(error)

def normalized_error_rate(a,b,raw_error):
    error = absolute(a-b) > 0
    return np.mean(error) / raw_error

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/256map

def DMC(x, PI):
    n = len(x)
    z = np.zeros(n, dtype = int)
    hid_states, obs_states = PI.shape[0], PI.shape[1]
    PI_sum = np.copy(PI)
    for i in range(1, obs_states):
        PI_sum.T[i] += PI_sum.T[i-1]
    prob = np.random.random()
    z[0] = int(np.argmax(PI_sum[x[0]] > prob))
    for i in range(1,n):
        prob = np.random.random()
        z[i] = int(np.argmax(PI_sum[x[i]] > prob))
    return z


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/256map

def L_NEW(PI, nb_x_classes, nb_z_classes):
    PI_INV = linalg.inv(PI)
    RHO = np.zeros((nb_x_classes, nb_z_classes ** nb_x_classes))
    LAMBDA = np.ones((nb_x_classes, nb_x_classes)) - np.eye(nb_x_classes)

    MAP = np.ones((nb_z_classes, nb_z_classes ** nb_x_classes), dtype = int)
    idx = 0
    for a in range(nb_z_classes):
        for b in range(nb_z_classes):
            for c in range(nb_z_classes):
                for d in range(nb_z_classes):
                    MAP[0][idx], MAP[1][idx], MAP[2][idx], MAP[3][idx] = a, b, c, d
    
    
    for i in range(nb_x_classes):
        RHO[i] = np.matmul(PI[i],LAMBDA[i][MAP])

    L = np.matmul(PI_INV, RHO)
    L_new = -L + amax(L)
    return L_new


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/256map

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


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/256map

def PREPROCESS(lines,nt_order):
    z    = zeros(2500000,dtype=int)
    zn = 0
    for t in range(len(lines)):
        if t % 2 == 0:
            continue

        for i in range(len(lines[t])-1):
            if zn == len(z):
                break
            if nt_order.find(lines[t][i]) < 0:
                z[zn] = random.randint(0,4)
                zn += 1
                continue
            for j in range(4):
                if lines[t][i] == nt_order[j]:
                    z[zn] = j
                    zn += 1
                    break

    return z[:zn]


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/256map

def load_DNA(PI_true):
    file_name = "16S_rRNA"
    f_in = open("data/Simluted_%s_Nanopore_x.fa" % file_name, "r")
    f_x = f_in.readlines()
    f_in.close()
    nb_classes=4
    nt_order = "ATGC"
    x = PREPROCESS(f_x,nt_order)
    z = DMC(x, PI_true)
    return x, z


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/256map

def sym_mat(states, prob):
    x = ones((states,states)) * (prob/(states-1))
    for i in range(states):
        x[i][i] = 1 - (states-1)*x[i][i]
    return x

def make_batch(z, k):
    # batch size & dimension length
    n = len(z)
    N, D = int(ceil(len(z)/1000)), 1000
    
    # slice concatenated sequence
    C = np.zeros((N, D+2*k, 4))
    
    # Convert 0,1,2,3 to one-hot vector
    LB = LabelBinarizer()
    LB.fit([0,1,2,3])
    for i in range(0,len(z),D):
        idx = int(i/D)
        diff = min([D,len(z)-i])
        C[idx,k:k+diff,:] = LB.transform(z[i:i+diff])
        
    return C

def make_pseudo_label(C, k, L_new):
    # batch size & dimension length
    N, D_, _ = C.shape
    D = D_-2*k
    Y = np.zeros((N, D, 256))
    
    for idx in range(N):
        Y[idx,:D,:] = dot(C[idx,k:k+D, :], L_new)
        
    return Y


# In[ ]:


# ICML_2019/DNA_CNN/Denoising/256map

def NDUDE_CNN_model_5map(D, nb_x_classes, nb_z_classes, k, lr = 0.001):
    unitN = 100
    # -----------------------------------------------------
    # Defining neural networks
    # -----------------------------------------------------
    inputs = layers.Input(shape = (D+2*k,nb_z_classes))
    layer = masked_CNN(unitN, 2*k+1, kernel_initializer = 'he_normal', padding='valid')(inputs)
    #layer = layers.Conv1D(unitN, 2*k+1, kernel_initializer = 'he_normal', padding='valid')(inputs)
    layer = layers.Activation('relu')(layer)
    layer = layers.Conv1D(unitN, 1, kernel_initializer = 'he_normal', padding='valid')(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Conv1D(unitN, 1, kernel_initializer = 'he_normal', padding='valid')(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Conv1D(nb_z_classes ** nb_x_classes, 1, kernel_initializer = 'he_normal', padding='valid')(layer)
    output = layers.Activation('softmax')(layer)
    model = models.Model(inputs = inputs, outputs = output)
    
    adam = optimizers.Adam(lr=lr)
    multi_model = multi_gpu_model(model, gpus=4)
    multi_model.compile(loss='poisson', optimizer=adam)
    return multi_model

