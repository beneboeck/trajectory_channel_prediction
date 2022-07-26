import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datetime
from utils import *
import dataset as ds
import training_unified as tr
import networks as mg
import evaluation_unified as ev
import math
import csv


# GLOBAL PARAMETERS
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 50
G_EPOCHS = 700
LEARNING_RATE = 6e-5
FREE_BITS_LAMBDA = torch.tensor(1).to(device)
SNAPSHOTS = 16
DATASET_TYPE = 'my_Quadriga'
MODEL_TYPE = 'Single' #Trajectory,Single
n_iterations = 75
n_permutations = 300
bs_mmd = 1000
normed = False
author = 'Bene'
SNR_db = 5


path_DFT_Tra = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_11_04_DFT_stable/model_dict'


# CREATING FILES AND DIRECTORY
now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
dir_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/evaluation/time_' + time
os.mkdir (dir_path)

glob_file = open(dir_path + '/glob_var_file.txt','w') # only the important results and the framework
log_file = open(dir_path + '/log_file.txt','w') # log_file which keeps track of the training and such stuff
csv_file = open(dir_path + '/csv_file.txt','w')
csv_writer = csv.writer(csv_file)
glob_file.write('Date: ' +date +'\n')
glob_file.write('Time: ' + time + '\n\n')
glob_file.write(f'\nMODEL_TYPE: {MODEL_TYPE}\n\n')
glob_file.write(f'\AUTHER: {author}\n\n')
glob_file.write('BATCHSIZE: ' + str(BATCHSIZE) +'\n')
glob_file.write('G_EPOCHS: ' +str(G_EPOCHS) +'\n')
glob_file.write(f'Learning Rate: {LEARNING_RATE}\n')
glob_file.write(f'SNR_db_list: {SNR_db}\n')
glob_file.write(f'n_iterations: {n_iterations}\n')
glob_file.write(f'n_permutations: {n_permutations}\n\n')
log_file.write('Date: ' +date +'\n')
log_file.write('Time: ' + time + '\n')
log_file.write('global variables successfully defined\n\n')
print('global var successful')

# DEFINING THE MODELS
cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'DFT',24,True,9,2,6,2,4,5,8,False,'DFT'
setup = [cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro]
model_DFT_Tra = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,device).to(device)
model_DFT_Tra.load_state_dict(torch.load(path_DFT_Tra,map_location=device))

model_DFT_Tra.eval()

H_test = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_test.npy','r')
H_train = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_train.npy','r')
H_val = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_val.npy','r')
pg_test = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/pg_test.npy','r')
pg_train = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/pg_train.npy','r')
pg_val = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/pg_val.npy','r')

H_test = H_test/np.sqrt(10**(0.1 * pg_test[:,None,None,0:1]))
H_val = H_val/np.sqrt(10**(0.1 * pg_val[:,None,None,0:1]))
H_train = H_train/np.sqrt(10**(0.1 * pg_train[:,None,None,0:1]))

        #print(np.mean(np.sum(np.abs(H_train)**2,axis=(1,2))))
        #print(np.mean(H_train))
        #print(np.std(H_train))

H_test_dft = apply_DFT(H_test)
H_val_dft = apply_DFT(H_val)
H_train_dft = apply_DFT(H_train)

x_train = np.mean(np.sum(H_train[:,:,:,-1]**2,axis=(1,2)))
SNR_eff = 10**(SNR_db/10)
sig_n_train = math.sqrt(x_train/(32 * SNR_eff))
x_test = np.mean(np.sum(H_test[:,:,:,-1]**2,axis=(1,2)))
SNR_eff = 10**(SNR_db/10)
sig_n_test = math.sqrt(x_test/(32 * SNR_eff))
x_val = np.mean(np.sum(H_val[:,:,:,-1]**2,axis=(1,2)))
SNR_eff = 10**(SNR_db/10)
sig_n_val = math.sqrt(x_val/(32 * SNR_eff))

n_H_train = H_train + sig_n_train/math.sqrt(2) * np.random.randn(*H_train.shape)
n_H_test = H_test + sig_n_test/math.sqrt(2) * np.random.randn(*H_test.shape)
n_H_val = H_val + sig_n_val/math.sqrt(2) * np.random.randn(*H_val.shape)
n_H_train_dft = H_train_dft + sig_n_train/math.sqrt(2) * np.random.randn(*H_train_dft.shape)
n_H_test_dft = H_test_dft + sig_n_test/math.sqrt(2) * np.random.randn(*H_test_dft.shape)
n_H_val_dft = H_val_dft + sig_n_val/math.sqrt(2) * np.random.randn(*H_val_dft.shape)

dataset_test = ds.dataset(H_test,H_test_dft,n_H_test, n_H_test_dft)
dataset_train = ds.dataset(H_train,H_train_dft,n_H_train, n_H_train_dft)
dataset_val = ds.dataset(H_val,H_val_dft,n_H_val, n_H_val_dft)

dataloader_test = DataLoader(dataset_test,shuffle=True,batch_size= len(dataset_test))
dataloader_train = DataLoader(dataset_train,shuffle=True,batch_size=BATCHSIZE)
dataloader_val = DataLoader(dataset_val,shuffle=True,batch_size= len(dataset_val))

NMSE_DFT_Tra = ev.channel_estimation(model_DFT_Tra, dataloader_test, sig_n_test, 'DFT', dir_path, device)

print(NMSE_DFT_Tra)