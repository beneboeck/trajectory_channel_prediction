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
from os.path import exists
import csv

# GLOBAL PARAMETERS
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 50
SNAPSHOTS = 16
MODEL_TYPE = 'Trajectory' #Trajectory,Single,TraSingle
SNR_db = 5

# CREATING FILES AND DIRECTORY
now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]

overall_path = '../Simulations/trajectory_channel_prediction/models/'
dir_path = '../Simulations/trajectory_channel_prediction/models/time_' + time
os.mkdir (dir_path)

glob_file = open(dir_path + '/glob_var_file.txt','w') # only the important results and the framework
log_file = open(dir_path + '/log_file.txt','w') # log_file which keeps track of the training and such stuff
glob_file.write('UPPER_BOUNDS OLD; LOWER BOUNDS NEW!!')
glob_file.write('Date: ' +date +'\n')
glob_file.write('Time: ' + time + '\n\n')
glob_file.write(f'\nMODEL_TYPE: {MODEL_TYPE}\n\n')
glob_file.write('BATCHSIZE: ' + str(BATCHSIZE) +'\n')
glob_file.write(f'SNR_db: {SNR_db}\n')
log_file.write('Date: ' +date +'\n')
log_file.write('Time: ' + time + '\n')
log_file.write('global variables successfully defined\n\n')
print('global var successful')

path_model = '../Simulations/trajectory_channel_prediction/models/time_10_37_myVAE/model_dict'

#Reproducing MODEL
LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type, BN, prepro,n_conv,cnn_bool = 14,10,False,3,8,3,9,5,8,'DFT',False,'DFT',1,False
# BEST DFT YET WITH NEW BOUNDS
#LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type, BN, prepro,n_conv,cnn_bool = 32,10,True,3,4,3,3,4,6,'DFT',False,'None',1,False
# FINAL TN MODEL
#LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type, BN, prepro,n_conv,cnn_bool = 32,10,False,3,4,3,3,4,6,'Toeplitz',False,'None',1,False
setup = [LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type, BN, prepro,n_conv,cnn_bool]

LD_VAE,conv_layer, tot_layer,out_channel,k_size,cov_type,prepro,LB,UB,BN = 56,0,3,128,7,'Toeplitz','None',0.01,0.6,False


H_test = np.load('../Simulations/trajectory_channel_prediction/data/H_test.npy','r')
H_train = np.load('../Simulations/trajectory_channel_prediction/data/H_train.npy','r')
H_val = np.load('../Simulations/trajectory_channel_prediction/data/H_val.npy','r')
pg_test = np.load('../Simulations/trajectory_channel_prediction/data/pg_test.npy','r')
pg_train = np.load('../Simulations/trajectory_channel_prediction/data/pg_train.npy','r')
pg_val = np.load('../Simulations/trajectory_channel_prediction/data/pg_val.npy','r')

H_test = H_test/np.sqrt(10**(0.1 * pg_test[:,None,None,0:1]))
H_val = H_val/np.sqrt(10**(0.1 * pg_val[:,None,None,0:1]))
H_train = H_train/np.sqrt(10**(0.1 * pg_train[:,None,None,0:1]))

print(np.mean(np.sum(np.abs(H_train)**2,axis=(1,2))))
print(np.mean(H_train))
print(np.std(H_train))

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

#model = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,device).to(device)
model = mg.my_VAE(cov_type,LD_VAE,conv_layer,tot_layer,out_channel,k_size,prepro,LB,UB,BN,device).to(device)
model.load_state_dict(torch.load(path_model,map_location=device))
model.eval()
NMSE_DFT_Tra = ev.channel_estimation('PERFECT',model, dataloader_test, sig_n_test, cov_type, dir_path, device)
print(NMSE_DFT_Tra)
NMSE, Risk,output_stats = ev.eval_val(MODEL_TYPE,setup,model, dataloader_val,cov_type, torch.tensor(1).to(device), device, dir_path)
print(output_stats)