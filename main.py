import torch
import sys
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import h5py

import datetime
import tqdm
from scipy import linalg as la
from sklearn.svm import SVC
import random
import umap


GLOBAL_ARCHITECTURE = 'Masked_HM_VAE'
# options: - 'genericVAE' - 'kalmanVAE' - 'genericGlow' - 'markovVAE' - 'hiddenMarkovVanillaVAE' -
#             'markovVanillaVAE' -'kMemoryHiddenMarkovVAE' - 'ApproxKMemoryHiddenMarkovVAE' -'kMemoryMarkovVAE' - 'WN_kMemoryHiddenMarkovVAE'
#             'WN_ModelBasedKMemoryHiddenMarkovVAE' , 'LSTM_HM_VAE', 'Masked_HM_VAE'

now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
dir_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_' + time + '_' + GLOBAL_ARCHITECTURE
os.mkdir (dir_path)
glob_var_file = open(dir_path + '/glob_var_file.txt','w')
log_file = open(dir_path + '/log_file.txt','w')
m_file = open(dir_path + '/m_file.txt','w')

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

BATCHSIZE = 50
G_EPOCHS = 700
LEARNING_RATE = 3e-5
STANDARDIZE_METHOD = 'c_s' # 'pg_s','c_s' (c_s: conventional standardization, pg_s: path gain standardization)
FREE_BITS_LAMBDA = torch.tensor(1).to(device) # is negligible if free bits isn't used
SNAPSHOTS = 32 # 96 / 192 should be taken for all models expect the modelbased one
DATASET_TYPE = 'Quadriga'
VELOCITY = 2

ARCHITECTURE_FAMILY = 'toeplitz'
# options: 'toeplitz' - 'unitary' - 'diagonal' - 'cholesky'
LOCAL_ARCHITECTURE = 'toeplitz'
# options for genericVAE: - 'toeplitz' - 'toeplitz_same_pre' - 'unitary' - 'unitary_same_pre' - 'diagonal' - 'diagonal_same_pre' - 'cholesky'
# options for kalmanVAE: - 'toeplitz' - 'toeplitz_same_dec' - 'toeplitz_same_all' - 'unitary' - 'unitary_same_dec' - 'unitary_same_all' - 'diagonal' - 'diagonal_same_dec' - 'diagonal_same_all'
# options for Glow: - ''


if (GLOBAL_ARCHITECTURE == 'kalmanVAE') | (GLOBAL_ARCHITECTURE == 'WN_kalmanVAE'):
    DIM_VEC = [([16],[2,32,SNAPSHOTS],1,device)]
    LATENT_DIMENSIONS,INTERNAL_DIMENSIONS,INPUT_SIZE = None,None,range(len(DIM_VEC))
    RP_NN,Y_max,Y_min = None,None,None
# DIM_VEC = LATENT DIM PER UNIT, DIMENSION OF WHOLE INPUT, NUMBER TIME STEPS PER UNIT
# some options: - [32],[2,32,100],4 - [16],[2,32,100],2 -[([32], [2, 32, 96], 4,device),([16],[2,32,96],2,device),([8],[2,32,96],2,device),([8],[2,32,96],1,device)]
# IMPORTANT: TIME STEPS / NUMBER TIME STEPS PER UNIT has to be an integer!
# IMPORTANT!! IT HAS TO BE A LIST

if (GLOBAL_ARCHITECTURE == 'kMemoryHiddenMarkovVAE') | (GLOBAL_ARCHITECTURE == 'WN_kMemoryHiddenMarkovVAE'):
    DIM_VEC = [([12],[2,32,SNAPSHOTS],1,6,device)]
    LATENT_DIMENSIONS,INTERNAL_DIMENSIONS,INPUT_SIZE = None,None,range(len(DIM_VEC))
    RP_NN,Y_max,Y_min = None,None,None
# DIM_VEC = LATENT DIM PER UNIT, DIMENSION OF WHOLE INPUT, NUMBER TIME STEPS PER UNIT, MEMORY
# some options: - [32],[2,32,100],4 - [16],[2,32,100],2 -[([32], [2, 32, 96], 4,device),([16],[2,32,96],2,device),([8],[2,32,96],2,device),([8],[2,32,96],1,device)]
# IMPORTANT: TIME STEPS / NUMBER TIME STEPS PER UNIT has to be an integer!
# IMPORTANT!! IT HAS TO BE A LIST

if GLOBAL_ARCHITECTURE == 'ApproxKMemoryHiddenMarkovVAE':
    DIM_VEC = [([8],[2,32,SNAPSHOTS],1,4,device)]
    LATENT_DIMENSIONS,INTERNAL_DIMENSIONS,INPUT_SIZE = None,None,range(len(DIM_VEC))
    RP_NN,Y_max,Y_min = None,None,None
# DIM_VEC = LATENT DIM PER UNIT, DIMENSION OF WHOLE INPUT, NUMBER TIME STEPS PER UNIT, MEMORY
# some options: - [32],[2,32,100],4 - [16],[2,32,100],2 -[([32], [2, 32, 96], 4,device),([16],[2,32,96],2,device),([8],[2,32,96],2,device),([8],[2,32,96],1,device)]
# IMPORTANT: TIME STEPS / NUMBER TIME STEPS PER UNIT has to be an integer!
# IMPORTANT!! IT HAS TO BE A LIST

RISK_TYPE = '_free_bits'
# options: - '_free_bits' - '' (for Glow only '' is possible)
RISK_TYPE = GLOBAL_ARCHITECTURE + '_' + ARCHITECTURE_FAMILY + RISK_TYPE


#WRITING INITIAL INFORMATIONS INTO THE FILES

glob_var_file.write('Date: ' +date +'\n')
glob_var_file.write('Time: ' + time + '\n')
glob_var_file.write('GLOBAL_ARCHITECTURE: ' + GLOBAL_ARCHITECTURE +'\n')
glob_var_file.write('LOCAL_ARCHITECTURE: ' + LOCAL_ARCHITECTURE + '\n')
if GLOBAL_ARCHITECTURE == 'kalmanVAE':
    glob_var_file.write('DIM_VEC: ' + str(DIM_VEC) + '\n')
glob_var_file.write('RISK_TYPE: ' +RISK_TYPE +'\n')
glob_var_file.write('BATCHSIZE: ' + str(BATCHSIZE) +'\n')
glob_var_file.write('G_EPOCHS: ' +str(G_EPOCHS) +'\n')
log_file.write('Date: ' +date +'\n')
log_file.write('Time: ' + time + '\n')
log_file.write('global variables successfully defined\n')
print('global variables successfully defined')

#LOADING AND PREPARING DATA + DEFINING THE MODEL

if DATASET_TYPE == 'Quadriga':
    label_test = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/label_test.npy','r')
    label_train = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/label_train.npy','r')
    label_val = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/label_val.npy','r')
    x_test = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/x_test.npy','r')
    x_train = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/x_train.npy','r')
    x_val = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/x_val.npy','r')
    y_test = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/y_test.npy','r')
    y_train = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/y_train.npy','r')
    y_val = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/Quadriga_Valentina/y_val.npy','r')


x_train = x_train[label_train == VELOCITY]
y_train = y_train[label_train == VELOCITY]
y_train = y_train[:,1:,:]

data = np.concatenate((x_train,y_train),axis=1)
print(data.shape)
print(np.mean(data[:,0,0]))
print(np.std(data[:,0,0]))

print(x_train.shape)
#print(y_train)
print(label_train[label_train == VELOCITY][:10])
print(x_train[0,-1,0] == y_train[0,0,0])