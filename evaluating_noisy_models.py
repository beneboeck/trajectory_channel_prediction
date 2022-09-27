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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
SNR_db_list = [-10,-5,0,5,10,20]


path_DFT_VAE_0dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_VAE_0dB/time_18_59_NMSE/model_dict'
path_DFT_VAE_5dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_VAE_5dB/time_21_10_NMSE/model_dict'
path_DFT_VAE_10dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_VAE_10dB/time_22_49_NMSE/model_dict'
path_DFT_VAE_20dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_VAE_20dB/time_13_20_NMSE/model_dict'

path_DFT_HMVAE_0dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_HMVAE_0dB/time_12_55_NMSE/model_dict'
path_DFT_HMVAE_5dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_HMVAE_5dB/time_17_12_NMSE/model_dict'
path_DFT_HMVAE_10dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_HMVAE_10dB/time_20_00_NMSE/model_dict'
path_DFT_HMVAE_20dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_HMVAE_20dB/time_01_29_NMSE/model_dict'

path_DFT_TraVAE_0dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_TraVAE_0dB/time_09_39_NMSE/model_dict'
path_DFT_TraVAE_5dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_TraVAE_5dB/time_21_13_NMSE/model_dict'
path_DFT_TraVAE_10dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_TraVAE_10dB/time_09_15_NMSE/model_dict'
path_DFT_TraVAE_20dB = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_TraVAE_20dB/time_10_16_NMSE/model_dict'

path_DFT_VAE_RANGE = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_VAE_RANGE/time_10_59/model_dict'
path_TraVAE_RANGE = '../trajectory_channel_prediction_data_analysis/HMVAE_DFT_paper/best_TraVAE_RANGE/time_11_03/model_dict'


# CREATING FILES AND DIRECTORY
now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
#dir_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/evaluation/time_' + time
dir_path = '../trajectory_channel_prediction_data_analysis/evaluation/time_' + time
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

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro,LB,UB,BN,reg_output_var = 40,3,3,128,9,'DFT','DFT',0.0020,0.5194,False,False
model_DFT_0dB = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,LB,UB,BN,reg_output_var,device).to(device)
model_DFT_0dB.load_state_dict(torch.load(path_DFT_VAE_0dB,map_location=device))

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro,LB,UB,BN,reg_output_var = 24,3,3,128,9,'DFT','DFT',0.0076,0.7297,False,True
model_DFT_5dB = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,LB,UB,BN,reg_output_var,device).to(device)
model_DFT_5dB.load_state_dict(torch.load(path_DFT_VAE_5dB,map_location=device))

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro,LB,UB,BN,reg_output_var = 16,2,4,128,7,'DFT','DFT',0.0032,0.8348,False,False
model_DFT_10dB = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,LB,UB,BN,reg_output_var,device).to(device)
model_DFT_10dB.load_state_dict(torch.load(path_DFT_VAE_10dB,map_location=device))

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro,LB,UB,BN,reg_output_var = 24,0,3,128,7,'DFT','DFT',0.0041,0.5286,True,False
model_DFT_20dB = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,LB,UB,BN,reg_output_var,device).to(device)
model_DFT_20dB.load_state_dict(torch.load(path_DFT_VAE_20dB,map_location=device))

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro,LB,UB,BN,reg_output_var = 24,3,3,128,9,'DFT','DFT',0.0076,0.7297,False,True
model_DFT_RANGE = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,LB,UB,BN,reg_output_var,device).to(device)
model_DFT_RANGE.load_state_dict(torch.load(path_DFT_VAE_RANGE,map_location=device))


cov_type,LD,conv_layer,total_layer,out_channel,k_size,LB,UB,BN,prepro,reg_output_var = 'DFT',256,0,4,128,7,0.0059,0.9278,False,'None',False
model_Tra_0dB = mg.my_tra_VAE(cov_type, LD, conv_layer, total_layer, out_channel, k_size, prepro,SNAPSHOTS,LB,UB,BN,reg_output_var,device).to(device)
model_Tra_0dB.load_state_dict(torch.load(path_DFT_TraVAE_0dB,map_location=device))

cov_type,LD,conv_layer,total_layer,out_channel,k_size,LB,UB,BN,prepro,reg_output_var = 'DFT',64,0,4,64,7,0.008,0.7708,False,'None',False
model_Tra_5dB = mg.my_tra_VAE(cov_type, LD, conv_layer, total_layer, out_channel, k_size, prepro,SNAPSHOTS,LB,UB,BN,reg_output_var,device).to(device)
model_Tra_5dB.load_state_dict(torch.load(path_DFT_TraVAE_5dB,map_location=device))

cov_type,LD,conv_layer,total_layer,out_channel,k_size,LB,UB,BN,prepro,reg_output_var = 'DFT',256,2,3,128,7,0.0038,0.8209,True,'None',False
model_Tra_10dB = mg.my_tra_VAE(cov_type, LD, conv_layer, total_layer, out_channel, k_size, prepro,SNAPSHOTS,LB,UB,BN,reg_output_var,device).to(device)
model_Tra_10dB.load_state_dict(torch.load(path_DFT_TraVAE_10dB,map_location=device))

cov_type,LD,conv_layer,total_layer,out_channel,k_size,LB,UB,BN,prepro,reg_output_var = 'DFT',128,2,3,64,7,0.009,0.5052,False,'None',False
model_Tra_20dB = mg.my_tra_VAE(cov_type, LD, conv_layer, total_layer, out_channel, k_size, prepro,SNAPSHOTS,LB,UB,BN,reg_output_var,device).to(device)
model_Tra_20dB.load_state_dict(torch.load(path_DFT_TraVAE_20dB,map_location=device))

cov_type,LD,conv_layer,total_layer,out_channel,k_size,LB,UB,BN,prepro,reg_output_var = 'DFT',64,0,4,64,7,0.008,0.7708,False,'None',False
model_Tra_RANGE = mg.my_tra_VAE(cov_type, LD, conv_layer, total_layer, out_channel, k_size, prepro,SNAPSHOTS,LB,UB,BN,reg_output_var,device).to(device)
model_Tra_RANGE.load_state_dict(torch.load(path_TraVAE_RANGE,map_location=device))


cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'DFT',24,False,8,2,9,3,6,4,6,False,'DFT'
n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var = 1,True,0.0086,0.7553,False
model_HMVAE_0dB = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var,device).to(device)
model_HMVAE_0dB.load_state_dict(torch.load(path_DFT_HMVAE_0dB,map_location=device))

cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'DFT',32,False,8,3,3,3,6,4,6,False,'DFT'
n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var = 2,True,0.0084,0.8965,True
model_HMVAE_5dB = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var,device).to(device)
model_HMVAE_5dB.load_state_dict(torch.load(path_DFT_HMVAE_5dB,map_location=device))

cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'DFT',32,True,4,2,9,3,8,5,6,False,'None'
n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var = 1,False,0.0041,0.6008,False
model_HMVAE_10dB = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var,device).to(device)
model_HMVAE_10dB.load_state_dict(torch.load(path_DFT_HMVAE_10dB,map_location=device))

cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'DFT',40,True,6,4,6,2,8,4,8,False,'DFT'
n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var = 2,True,0.0017,0.5235,False
model_HMVAE_20dB = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var,device).to(device)
model_HMVAE_20dB.load_state_dict(torch.load(path_DFT_HMVAE_20dB,map_location=device))


model_DFT_0dB.eval()
model_DFT_5dB.eval()
model_DFT_10dB.eval()
model_DFT_20dB.eval()

model_Tra_0dB.eval()
model_Tra_5dB.eval()
model_Tra_10dB.eval()
model_Tra_20dB.eval()

model_HMVAE_0dB.eval()
model_HMVAE_5dB.eval()
model_HMVAE_10dB.eval()
model_HMVAE_20dB.eval()

model_DFT_RANGE.eval()
model_Tra_RANGE.eval()

models_VAE = [model_DFT_5dB,model_DFT_5dB,model_DFT_5dB,model_DFT_5dB,model_DFT_5dB,model_DFT_5dB]
models_Tra = [model_Tra_5dB,model_Tra_5dB,model_Tra_5dB,model_Tra_5dB,model_Tra_5dB,model_Tra_5dB]
models_HMVAE = [model_HMVAE_5dB,model_HMVAE_5dB,model_HMVAE_5dB,model_HMVAE_5dB,model_HMVAE_5dB,model_HMVAE_5dB]



csv_writer.writerow(SNR_db_list)

NMSE_VAE = []
NMSE_Tra = []
#NMSE_HMVAE = []
NMSE_LS_list = []
NMSE_sCov_list = []

for idx,SNR_db in enumerate(SNR_db_list):

    print(f'SNR_db: {SNR_db}')
    H_test_c = np.load('../Simulations/trajectory_channel_prediction/data/H_test_Uma_mixed_IO_600_200.npy', 'r')
    H_train_c = np.load('../Simulations/trajectory_channel_prediction/data/H_train_Uma_mixed_IO_600_200.npy', 'r')
    H_val_c = np.load('../Simulations/trajectory_channel_prediction/data/H_val_Uma_mixed_IO_600_200.npy', 'r')
    H_train = np.zeros((100000, 2, 32, 16))
    H_train[:, 0, :, :] = np.real(H_train_c)
    H_train[:, 1, :, :] = np.imag(H_train_c)

    H_val = np.zeros((10000, 2, 32, 16))
    H_val[:, 0, :, :] = np.real(H_val_c)
    H_val[:, 1, :, :] = np.imag(H_val_c)

    H_test = np.zeros((10000, 2, 32, 16))
    H_test[:, 0, :, :] = np.real(H_test_c)
    H_test[:, 1, :, :] = np.imag(H_test_c)

    print('....')
    print(np.mean(np.sum(np.abs(H_train) ** 2, axis=(1, 2))))
    print(np.mean(H_train))
    print(np.std(H_train))

    H_test_dft = apply_DFT(H_test)
    H_val_dft = apply_DFT(H_val)
    H_train_dft = apply_DFT(H_train)

    x_train = np.sum(H_train[:, :, :, -1] ** 2, axis=(1, 2))
    SNR_eff = 10 ** (SNR_db / 10)
    sig_n_train = np.sqrt(x_train / (32 * SNR_eff))[:, None, None, None]
    x_test = np.sum(H_test[:, :, :, -1] ** 2, axis=(1, 2))
    SNR_eff = 10 ** (SNR_db / 10)
    sig_n_test = np.sqrt(x_test / (32 * SNR_eff))[:, None, None, None]
    x_val = np.sum(H_val[:, :, :, -1] ** 2, axis=(1, 2))
    SNR_eff = 10 ** (SNR_db / 10)
    sig_n_val = np.sqrt(x_val / (32 * SNR_eff))[:, None, None, None]

    n_H_train = H_train + sig_n_train / math.sqrt(2) * np.random.randn(*H_train.shape)
    n_H_test = H_test + sig_n_test / math.sqrt(2) * np.random.randn(*H_test.shape)
    n_H_val = H_val + sig_n_val / math.sqrt(2) * np.random.randn(*H_val.shape)
    n_H_train_dft = H_train_dft + sig_n_train / math.sqrt(2) * np.random.randn(*H_train_dft.shape)
    n_H_test_dft = H_test_dft + sig_n_test / math.sqrt(2) * np.random.randn(*H_test_dft.shape)
    n_H_val_dft = H_val_dft + sig_n_val / math.sqrt(2) * np.random.randn(*H_val_dft.shape)

    dataset_test = ds.dataset(H_test, H_test_dft, n_H_test, n_H_test_dft, sig_n_test)
    dataset_train = ds.dataset(H_train, H_train_dft, n_H_train, n_H_train_dft, sig_n_train)
    dataset_val = ds.dataset(H_val, H_val_dft, n_H_val, n_H_val_dft, sig_n_val)

    dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=len(dataset_test))
    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=BATCHSIZE)
    dataloader_val = DataLoader(dataset_val, shuffle=True, batch_size=len(dataset_val))

    NMSE_VAE_one = ev.channel_estimation('NOISY',model_DFT_RANGE, dataloader_test, sig_n_test, 'DFT', dir_path, device)[0]
    NMSE_Tra_one = ev.channel_estimation('NOISY',model_Tra_RANGE, dataloader_test, sig_n_test, 'DFT', dir_path, device)[0]
    #NMSE_HMVAE_one = ev.channel_estimation('NOISY',models_HMVAE[idx], dataloader_test, sig_n_test, 'DFT', dir_path, device)[0]


    #NMSE_LS_tot, NMSE_sCov_tot = ev.computing_LS_sample_covariance_estimator_all(dataset_test, sig_n_test)
    NMSE_LS,NMSE_sCov = ev.computing_LS_sample_covariance_estimator(dataset_test,dataset_train,sig_n_test)

    NMSE_VAE.append(NMSE_VAE_one)
    NMSE_Tra.append(NMSE_Tra_one)
    #NMSE_HMVAE.append(NMSE_HMVAE_one)
    NMSE_LS_list.append(NMSE_LS.item())
    NMSE_sCov_list.append(NMSE_sCov.item())

#csv_writer.writerow(NMSE_est_DFT_Tra)
#csv_writer.writerow(NMSE_est_TD_Tra)
#csv_writer.writerow(NMSE_est_TN_Tra)

csv_writer.writerow(NMSE_VAE)
csv_writer.writerow(NMSE_Tra)
#csv_writer.writerow(NMSE_HMVAE)
csv_writer.writerow(NMSE_LS_list)
csv_writer.writerow(NMSE_sCov_list)
csv_file.close()
