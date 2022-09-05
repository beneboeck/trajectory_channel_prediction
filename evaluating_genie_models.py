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
SNR_db_list = [-10,-5,0,5,10,15,20]


path_DFT_Tra = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/first_run_models/time_17_46_tra/model_dict'
path_TN_Tra = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_18_23_final_TN/model_dict'
#path_TD_Tra = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/first_run_models/time_17_08_tra/model_dict'
path_DFT_VAE = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/first_run_models/time_20_25_single/model_dict'
path_TN_VAE = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/first_run_models/time_23_28_single/model_dict'
#path_TD_VAE = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/first_run_models/time_20_06_single/model_dict'


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
glob_file.write(f'SNR_db_list: {SNR_db_list}\n')
glob_file.write(f'n_iterations: {n_iterations}\n')
glob_file.write(f'n_permutations: {n_permutations}\n\n')
log_file.write('Date: ' +date +'\n')
log_file.write('Time: ' + time + '\n')
log_file.write('global variables successfully defined\n\n')
print('global var successful')

# DEFINING THE MODELS
cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'DFT',14,True,10,3,9,3,8,5,8,False,'None'
n_conv,cnn_bool,LB_var_dec,UB_var_dec = 2,False,0.1,0.33
model_DFT_Tra = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,device).to(device)
model_DFT_Tra.load_state_dict(torch.load(path_DFT_Tra,map_location=device))

cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'Toeplitz',32,False,10,3,3,3,4,4,6,False,'None'
n_conv,cnn_bool,LB_var_dec,UB_var_dec = 1,False,0,1
model_TN_Tra = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,device).to(device)
model_TN_Tra.load_state_dict(torch.load(path_TN_Tra,map_location=device))

#cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'Toeplitz',14,True,6,2,3,3,4,5,8,False,'DFT'
#model_TD_Tra = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,device).to(device)
#model_TD_Tra.load_state_dict(torch.load(path_TD_Tra,map_location=device))

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro = 56,3,4,128,9,'DFT','None'
model_DFT_VAE = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,device).to(device)
model_DFT_VAE.load_state_dict(torch.load(path_DFT_VAE,map_location=device))

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro = 40,3,3,128,7,'Toeplitz','None'
model_TN_VAE = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,device).to(device)
model_TN_VAE.load_state_dict(torch.load(path_TN_VAE,map_location=device))

#LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro = 48,3,5,128,9,'Toeplitz','DFT'
#model_TD_VAE = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,device).to(device)
#model_TD_VAE.load_state_dict(torch.load(path_TD_VAE,map_location=device))

#model_TD_VAE.eval()
model_TN_VAE.eval()
model_DFT_VAE.eval()
#model_TD_Tra.eval()
model_TN_Tra.eval()
model_DFT_Tra.eval()

csv_writer.writerow(SNR_db_list)

NMSE_est_DFT_Tra = []
NMSE_est_TD_Tra = []
NMSE_est_TN_Tra = []

NMSE_est_DFT_VAE = []
NMSE_est_TD_VAE = []
NMSE_est_TN_VAE = []
NMSE_est_DFT_VAE_tot = []
NMSE_est_TD_VAE_tot = []
NMSE_est_TN_VAE_tot = []

NMSE_est_LS = []
NMSE_est_LS_tot = []
NMSE_est_sCov = []
NMSE_est_sCov_tot = []


for SNR_db in SNR_db_list:

    if DATASET_TYPE == 'my_Quadriga':
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
    #NMSE_TD_Tra = ev.channel_estimation(model_TD_Tra, dataloader_test, sig_n_test, 'Toeplitz', dir_path, device)
    NMSE_TN_Tra = ev.channel_estimation(model_TN_Tra, dataloader_test, sig_n_test, 'Toeplitz', dir_path, device)

    NMSE_DFT_VAE = ev.channel_estimation(model_DFT_VAE, dataloader_test, sig_n_test, 'DFT', dir_path, device)
    NMSE_DFT_VAE_tot = ev.channel_estimation_all(model_DFT_VAE, dataloader_test, sig_n_test, 'DFT', dir_path, device)
    #NMSE_TD_VAE = ev.channel_estimation(model_TD_VAE, dataloader_test, sig_n_test, 'Toeplitz', dir_path, device)
    #NMSE_TD_VAE_tot = ev.channel_estimation_all(model_TD_VAE, dataloader_test, sig_n_test, 'Toeplitz', dir_path, device)
    NMSE_TN_VAE = ev.channel_estimation(model_TN_VAE, dataloader_test, sig_n_test, 'Toeplitz', dir_path, device)
    NMSE_TN_VAE_tot = ev.channel_estimation_all(model_TN_VAE, dataloader_test, sig_n_test, 'Toeplitz', dir_path, device)

    NMSE_LS_tot, NMSE_sCov_tot = ev.computing_LS_sample_covariance_estimator_all(dataset_test, sig_n_test)
    NMSE_LS, NMSE_sCov = ev.computing_LS_sample_covariance_estimator(dataset_test, sig_n_test)

    NMSE_est_DFT_Tra.append(NMSE_DFT_Tra)
    #NMSE_est_TD_Tra.append(NMSE_TD_Tra)
    NMSE_est_TN_Tra.append(NMSE_TN_Tra)

    NMSE_est_DFT_VAE.append(NMSE_DFT_VAE)
    #NMSE_est_TD_VAE.append(NMSE_TD_VAE)
    NMSE_est_TN_VAE.append(NMSE_TN_VAE)
    NMSE_est_DFT_VAE_tot.append(NMSE_DFT_VAE_tot)
    #NMSE_est_TD_VAE_tot.append(NMSE_TD_VAE_tot)
    NMSE_est_TN_VAE_tot.append(NMSE_TN_VAE_tot)

    NMSE_est_LS.append(NMSE_LS.item())
    NMSE_est_LS_tot.append(NMSE_LS_tot.item())
    NMSE_est_sCov.append(NMSE_sCov.item())
    NMSE_est_sCov_tot.append(NMSE_sCov_tot.item())

csv_writer.writerow(NMSE_est_DFT_Tra)
csv_writer.writerow(NMSE_est_TD_Tra)
csv_writer.writerow(NMSE_est_TN_Tra)

csv_writer.writerow(NMSE_est_DFT_VAE)
csv_writer.writerow(NMSE_est_TD_VAE)
csv_writer.writerow(NMSE_est_TN_VAE)
csv_writer.writerow(NMSE_est_DFT_VAE_tot)
csv_writer.writerow(NMSE_est_TD_VAE_tot)
csv_writer.writerow(NMSE_est_TN_VAE_tot)

csv_writer.writerow(NMSE_est_LS)
csv_writer.writerow(NMSE_est_LS_tot)
csv_writer.writerow(NMSE_est_sCov)
csv_writer.writerow(NMSE_est_sCov_tot)

csv_file.close()
