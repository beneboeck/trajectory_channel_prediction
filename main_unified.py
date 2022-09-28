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

################################################# GLOBAL PARAMETERS ############################################################
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 50
G_EPOCHS = 600
LEARNING_RATE = 6e-5
FREE_BITS_LAMBDA = torch.tensor(0.1).to(device)
SNAPSHOTS = 16
MODEL_TYPE = 'TraSingle' #Trajectory,Single,TraSingle
n_iterations = 75
n_permutations = 300
bs_mmd = 1000
normed = False
SNR_db = 5
SNR_range = [-10,25]
CSI = 'NOISY' # PERFECT, NOISY
SNR_format = 'RANGE' # RANGE,NO
RECURRENT = False

################################################ CREATING FILES AND DIRECTORY #############################################################
now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]

#overall_path = './'
#dir_path = './time_' + time

overall_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/'
dir_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_' + time
os.mkdir (dir_path)

if not(exists(overall_path + MODEL_TYPE + '_' + 'RANGE_noise_NAS_file.txt')):
    csvfile = open(overall_path + MODEL_TYPE + '_' + 'RANGE_noise_NAS_file.txt','w')
    csv_writer = csv.writer(csvfile)
    if MODEL_TYPE == 'Trajectory':
        csv_writer.writerow(['Time','LD', 'memory', 'rnn_bool', 'en_layer', 'en_width', 'pr_layer', 'pr_width', 'de_layer', 'de_width', 'cov_type', 'BN', 'prepro','DecVarLB','DecVarUB','TPR','TPRinf','Risk_val','NMSE_0dB','NMSE_5dB','NMSE_10dB','NMSE_20dB'])
    if (MODEL_TYPE == 'Single') | (MODEL_TYPE == 'TraSingle'):
        csv_writer.writerow(['Time','LD_VAE', 'conv_layer', 'total_layer', 'out_channel', 'k_size', 'cov_type','prepro','LB_var_dec','UB_var_dec','BN','Risk_val','NMSE_0dB','NMSE_5dB','NMSE_10dB','NMSE_20dB'])
    csvfile.close()

glob_file = open(dir_path + '/glob_var_file.txt','w') # only the important results and the framework
log_file = open(dir_path + '/log_file.txt','w') # log_file which keeps track of the training and such stuff
glob_file.write('NOISY ADDED IN EVERY EPOCH NEWLY')
glob_file.write('Date: ' +date +'\n')
glob_file.write('Time: ' + time + '\n\n')
glob_file.write(f'CSI TYPE: {CSI}\n\n')
glob_file.write(f'\nMODEL_TYPE: {MODEL_TYPE}\n\n')
glob_file.write('BATCHSIZE: ' + str(BATCHSIZE) +'\n')
glob_file.write('G_EPOCHS: ' +str(G_EPOCHS) +'\n')
glob_file.write(f'Learning Rate: {LEARNING_RATE}\n')
glob_file.write(f'SNR_db: {SNR_db}\n')
glob_file.write(f'n_iterations: {n_iterations}\n')
glob_file.write(f'n_permutations: {n_permutations}\n\n')
glob_file.write(f'SNR_format: {SNR_format}\n')
glob_file.write(f'RECURRENT: {RECURRENT}\n')
log_file.write('Date: ' +date +'\n')
log_file.write('Time: ' + time + '\n')
log_file.write('global variables successfully defined\n\n')
print('global var successful')

############################################### NETWORK ARCHITECTURE SEARCH #############################################
if MODEL_TYPE == 'Trajectory':
    LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var = network_architecture_search()
    rnn_bool = False
    setup = [LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type,BN,prepro,n_conv,cnn_bool]
    print('Trajectory Setup')
    print(LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var)
    glob_file.write(f'Latent Dim: {LD}\n')
    glob_file.write(f'Memory: {memory}\n')
    glob_file.write(f'RNN Bool: {rnn_bool}\n')
    glob_file.write(f'En_Layer: {en_layer}\n')
    glob_file.write(f'En_Width: {en_width}\n')
    glob_file.write(f'Pr_Layer: {pr_layer}\n')
    glob_file.write(f'Pr_Width: {pr_width}\n')
    glob_file.write(f'De_Layer: {de_layer}\n')
    glob_file.write(f'De_Width: {de_width}\n')
    glob_file.write(f'Cov_Type: {cov_type}\n')
    glob_file.write(f'n_conv: {n_conv}\n')
    glob_file.write(f'cnn_bool: {cnn_bool}\n')
    glob_file.write(f'BN use: {BN}\n')
    glob_file.write(f'preopro: {prepro}\n')
    glob_file.write(f'LB_var_dec: {LB_var_dec:.4f}\n')
    glob_file.write(f'UB_var_dec: {UB_var_dec:.4f}\n')
    glob_file.write(f'reg_output_var: {reg_output_var}\n')
if MODEL_TYPE == 'Single':
    LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro,LB_var_dec,UB_var_dec,BN,reg_output_var = network_architecture_search_VAE()
    setup = [LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro,reg_output_var]
    print('Single Setup')
    print(LD_VAE,conv_layer,total_layer,out_channel,k_size,cov_type,prepro,LB_var_dec,UB_var_dec,BN,reg_output_var)
    glob_file.write(f'\nlatent Dim VAE: {LD_VAE}\n')
    glob_file.write(f'conv_layer: {conv_layer}\n')
    glob_file.write(f'total_layer: {total_layer}\n')
    glob_file.write(f'out_channel: {out_channel}\n')
    glob_file.write(f'k_size: {k_size}\n')
    glob_file.write(f'cov_type: {cov_type}\n')
    glob_file.write(f'prepro: {prepro}\n')
    glob_file.write(f'LB_var_dec: {LB_var_dec:.4f}\n')
    glob_file.write(f'UB_var_dec: {UB_var_dec:.4f}\n')
    glob_file.write(f'BN: {BN:.4f}\n')
    glob_file.write(f'reg_output_var: {reg_output_var}\n')

if MODEL_TYPE == 'TraSingle':
    LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro,LB_var_dec,UB_var_dec,BN,reg_output_var = network_architecture_search_TraVAE()
    setup = [LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro]
    print('Single Setup')
    print(LD_VAE,conv_layer,total_layer,out_channel,k_size,cov_type,prepro,LB_var_dec,UB_var_dec,BN,reg_output_var)
    glob_file.write(f'\nlatent Dim VAE: {LD_VAE}\n')
    glob_file.write(f'conv_layer: {conv_layer}\n')
    glob_file.write(f'total_layer: {total_layer}\n')
    glob_file.write(f'out_channel: {out_channel}\n')
    glob_file.write(f'k_size: {k_size}\n')
    glob_file.write(f'cov_type: {cov_type}\n')
    glob_file.write(f'LB_var_dec: {LB_var_dec}\n')
    glob_file.write(f'UB_var_dec: {UB_var_dec}\n')
    glob_file.write(f'BN: {BN}\n')
    glob_file.write(f'prepro: {prepro}\n')
    glob_file.write(f'reg_output_var: {reg_output_var}\n')

#################################################################### LOADING AND PREPARING DATA FOR TRAINING #################################################

H_test_c = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_test_Uma_mixed_IO_600_200.npy','r')
H_train_c = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_train_Uma_mixed_IO_600_200.npy','r')
H_val_c = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_val_Uma_mixed_IO_600_200.npy','r')


#H_train_c = np.load('../../Projects/Simulations/trajectory_channel_prediction/data/H_train_Uma_mixed_IO_600_200.npy','r')
#H_test_c = np.load('../../Projects/Simulations/trajectory_channel_prediction/data/H_test_Uma_mixed_IO_600_200.npy','r')
#H_val_c = np.load('../../Projects/Simulations/trajectory_channel_prediction/data/H_val_Uma_mixed_IO_600_200.npy','r')

H_train = np.zeros((100000,2,32,16))
H_train[:,0,:,:] = np.real(H_train_c)
H_train[:,1,:,:] = np.imag(H_train_c)

H_val = np.zeros((10000,2,32,16))
H_val[:,0,:,:] = np.real(H_val_c)
H_val[:,1,:,:] = np.imag(H_val_c)

H_test = np.zeros((10000,2,32,16))
H_test[:,0,:,:] = np.real(H_test_c)
H_test[:,1,:,:] = np.imag(H_test_c)

print('....')
print(np.mean(np.sum(np.abs(H_train)**2,axis=(1,2))))
print(np.mean(H_train))
print(np.std(H_train))

H_test_dft = apply_DFT(H_test)
H_val_dft = apply_DFT(H_val)
H_train_dft = apply_DFT(H_train)

x_train = np.sum(H_train[:,:,:,-1]**2,axis=(1,2))
SNR_eff = 10**(SNR_db/10)
sig_n_train = np.sqrt(x_train/(32 * SNR_eff))[:,None,None,None]
x_test = np.sum(H_test[:,:,:,-1]**2,axis=(1,2))
SNR_eff = 10**(SNR_db/10)
sig_n_test = np.sqrt(x_test/(32 * SNR_eff))[:,None,None,None]
x_val = np.sum(H_val[:,:,:,-1]**2,axis=(1,2))
SNR_eff = 10**(SNR_db/10)
sig_n_val = np.sqrt(x_val/(32 * SNR_eff))[:,None,None,None]

n_H_train = H_train + sig_n_train/math.sqrt(2) * np.random.randn(*H_train.shape)
n_H_test = H_test + sig_n_test/math.sqrt(2) * np.random.randn(*H_test.shape)
n_H_val = H_val + sig_n_val/math.sqrt(2) * np.random.randn(*H_val.shape)
n_H_train_dft = H_train_dft + sig_n_train/math.sqrt(2) * np.random.randn(*H_train_dft.shape)
n_H_test_dft = H_test_dft + sig_n_test/math.sqrt(2) * np.random.randn(*H_test_dft.shape)
n_H_val_dft = H_val_dft + sig_n_val/math.sqrt(2) * np.random.randn(*H_val_dft.shape)

dataset_test = ds.dataset(H_test,H_test_dft,n_H_test, n_H_test_dft,sig_n_test)
dataset_train = ds.dataset(H_train,H_train_dft,n_H_train, n_H_train_dft,sig_n_train)
dataset_val = ds.dataset(H_val,H_val_dft,n_H_val, n_H_val_dft,sig_n_val)

dataloader_test = DataLoader(dataset_test,shuffle=True,batch_size= len(dataset_test))
dataloader_train = DataLoader(dataset_train,shuffle=True,batch_size=BATCHSIZE)
dataloader_val = DataLoader(dataset_val,shuffle=True,batch_size= len(dataset_val))

NMSE_LS,NMSE_sCov = ev.computing_LS_sample_covariance_estimator(dataset_val,dataset_train,sig_n_val)
print(f'LS,sCov estimation NMSE: {NMSE_LS:.4f},{NMSE_sCov:.4f}')


####################################################### CREATING THE MODELS & TRAINING #############################################
if MODEL_TYPE == 'Trajectory':
    if RECURRENT:
        model = mg.HMVAE_recurrent(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec,reg_output_var,device).to(device)
    else:
        model = mg.HMVAE(cov_type, LD, rnn_bool, 32, memory, pr_layer, pr_width, en_layer, en_width, de_layer,de_width, SNAPSHOTS, BN, prepro, n_conv, cnn_bool, LB_var_dec, UB_var_dec,reg_output_var, device).to(device)
if MODEL_TYPE == 'Single':
    model = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,LB_var_dec,UB_var_dec,BN,reg_output_var,device).to(device)
if MODEL_TYPE == 'TraSingle':
    model = mg.my_tra_VAE(cov_type, LD_VAE, conv_layer, total_layer, out_channel, k_size, prepro,SNAPSHOTS,LB_var_dec,UB_var_dec,BN,reg_output_var, device).to(device)
    print('model generated')

risk_list,KL_list,RR_list,eval_risk,eval_NMSE, eval_NMSE_estimation, eval_TPR1,eval_TPR2 = tr.training_gen_NN(SNR_format,SNR_range,CSI,MODEL_TYPE,setup,LEARNING_RATE,cov_type, model, dataloader_train,dataloader_val, G_EPOCHS, FREE_BITS_LAMBDA,sig_n_val,sig_n_train,device, log_file,dir_path,n_iterations, n_permutations, normed,bs_mmd, dataset_val, SNAPSHOTS)

################################################### EVALUATION OF THE MODELS #####################################################

model.eval()
save_risk(risk_list,RR_list,KL_list,dir_path,'Risks')

save_risk_single(eval_risk,dir_path,'Evaluation - ELBO')
save_risk_single(eval_NMSE,dir_path,'Evaluation - NMSE prediction')
save_risk_single(eval_NMSE_estimation,dir_path,'Evaluation - NMSE estimation')
save_risk_single(eval_TPR1,dir_path,'Evaluation - TPR1 prior')
save_risk_single(eval_TPR2,dir_path,'Evaluation - TPR2 - inference')

torch.save(model.state_dict(),dir_path + '/model_dict')


_, Risk_val, output_stats_val = ev.eval_val(CSI, MODEL_TYPE, setup, model, dataloader_val, cov_type, FREE_BITS_LAMBDA,device, dir_path)
if MODEL_TYPE == 'Trajectory':
    if cov_type == 'Toeplitz':
        m_sigma_squared_prior, std_sigma_squared_prior, m_sigma_squared_inf, std_sigma_squared_inf, m_alpha_0, std_alpha_0, n_bound_hits = output_stats_val
    if cov_type == 'DFT':
        m_sigma_squared_prior, std_sigma_squared_prior, m_sigma_squared_inf, std_sigma_squared_inf, m_sigma_squared_out, std_sigma_squared_out = output_stats_val
        m_alpha_0 = float('nan')
        n_bound_hits = float('nan')
    NMSE_test = ev.channel_prediction(CSI, setup, model, dataloader_test, 15, dir_path, device, 'testing')
    TPR1, TPR2 = ev.computing_MMD(CSI, setup, model, n_iterations, n_permutations, normed, bs_mmd, dataset_test,SNAPSHOTS, dir_path, device)
    NMSE_val = ev.channel_prediction(CSI, setup, model, dataloader_val, 15, dir_path, device, 'testing')
    TPR1_val, TPR2_val = ev.computing_MMD(CSI, setup, model, n_iterations, n_permutations, normed, bs_mmd, dataset_val,SNAPSHOTS, dir_path, device)

SNR_db_list = [0,5,10,20]
NMSE_est = []
for SNR_db in SNR_db_list:

  #  H_test = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_test.npy','r')
  #  H_train = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_train.npy','r')
  #  H_val = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/H_val.npy','r')
  #  pg_test = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/pg_test.npy','r')
  #  pg_train = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/pg_train.npy','r')
  #  pg_val = np.load('/home/ga42kab/lrz-nashome/trajectory_channel_prediction/data/my_quadriga/pg_val.npy','r')

  #  H_test = H_test/np.sqrt(10**(0.1 * pg_test[:,None,None,0:1]))
  #  H_val = H_val/np.sqrt(10**(0.1 * pg_val[:,None,None,0:1]))
  #  H_train = H_train/np.sqrt(10**(0.1 * pg_train[:,None,None,0:1]))

  #  H_test_dft = apply_DFT(H_test)
  #  H_val_dft = apply_DFT(H_val)
  #  H_train_dft = apply_DFT(H_train)

    x_train = np.sum(H_train[:, :, :, -1] ** 2, axis=(1, 2))
    SNR_eff = 10 ** (SNR_db / 10)
    sig_n_train = np.sqrt(x_train / (32 * SNR_eff))[:, None, None, None]
    x_test = np.sum(H_test[:, :, :, -1] ** 2, axis=(1, 2))
    SNR_eff = 10 ** (SNR_db / 10)
    sig_n_test = np.sqrt(x_test / (32 * SNR_eff))[:, None, None, None]
    x_val = np.sum(H_val[:, :, :, -1] ** 2, axis=(1, 2))
    SNR_eff = 10 ** (SNR_db / 10)
    sig_n_val = np.sqrt(x_val / (32 * SNR_eff))[:, None, None, None]

    n_H_train = H_train + sig_n_train/math.sqrt(2) * np.random.randn(*H_train.shape)
    n_H_test = H_test + sig_n_test/math.sqrt(2) * np.random.randn(*H_test.shape)
    n_H_val = H_val + sig_n_val/math.sqrt(2) * np.random.randn(*H_val.shape)
    n_H_train_dft = H_train_dft + sig_n_train/math.sqrt(2) * np.random.randn(*H_train_dft.shape)
    n_H_test_dft = H_test_dft + sig_n_test/math.sqrt(2) * np.random.randn(*H_test_dft.shape)
    n_H_val_dft = H_val_dft + sig_n_val/math.sqrt(2) * np.random.randn(*H_val_dft.shape)

    dataset_test = ds.dataset(H_test, H_test_dft, n_H_test, n_H_test_dft, sig_n_test)
    dataset_train = ds.dataset(H_train, H_train_dft, n_H_train, n_H_train_dft, sig_n_train)
    dataset_val = ds.dataset(H_val, H_val_dft, n_H_val, n_H_val_dft, sig_n_val)

    dataloader_test = DataLoader(dataset_test,shuffle=True,batch_size= len(dataset_test))
    dataloader_train = DataLoader(dataset_train,shuffle=True,batch_size=BATCHSIZE)
    dataloader_val = DataLoader(dataset_val,shuffle=True,batch_size= len(dataset_val))

    NMSE_val_est,mean_frob,mean_mu_signal_energy,Cov_part_LMMSE_energy,NMSE_only_mun = ev.channel_estimation(CSI,model,dataloader_val,sig_n_val,cov_type,dir_path,device)

    NMSE_est.append(NMSE_val_est)



csv_file = open(overall_path + MODEL_TYPE + '_' + 'RANGE_noise_NAS_file.txt','a')
csv_writer = csv.writer(csv_file)
if MODEL_TYPE == 'Trajectory':
    csv_writer.writerow([time,LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type, BN, prepro,LB_var_dec,UB_var_dec,TPR1_val,TPR2_val,round(Risk_val.item(),3),round(NMSE_est[0],5),round(NMSE_est[1],5),round(NMSE_est[2],5),round(NMSE_est[3],5)])
if (MODEL_TYPE == 'Single') | (MODEL_TYPE == 'TraSingle'):
    csv_writer.writerow([time,LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro,LB_var_dec,UB_var_dec,BN,round(Risk_val.item(),3),round(NMSE_est[0],5),round(NMSE_est[1],5),round(NMSE_est[2],5),round(NMSE_est[3],5)])

csv_file.close()


NMSE_LS,NMSE_sCov = ev.computing_LS_sample_covariance_estimator(dataset_val,dataset_train,sig_n_val)
print(f'LS,sCov estimation NMSE: {NMSE_LS:.4f},{NMSE_sCov:.4f}')
log_file.write(f'LS,sCov estimation NMSE: {NMSE_LS:.4f},{NMSE_sCov:.4f}\n')

glob_file.write('\nResults\n\n')
glob_file.write('EVALUATION SET\n\n')
glob_file.write(f'NMSE estimation: {eval_NMSE_estimation[-1]:.4f}\n')
glob_file.write(f'NMSE prediction: {eval_NMSE[-1]:.4f}\n')
glob_file.write(f'TPR - prior: {eval_TPR1[-1]:.4f}\n')
glob_file.write(f'TPR - inference: {eval_TPR2[-1]:.4f}\n')
glob_file.write(f'ELBO Validation Set: {Risk_val:4f}\n')
glob_file.write(f'Mean Frobenius Norm Cov: {mean_frob:.4f}\n')
glob_file.write(f'Mean MuOut Signal Energy: {mean_mu_signal_energy:.4f}\n')
glob_file.write(f'Mean CovLMMSE part Energy: {Cov_part_LMMSE_energy:.4f}\n')
glob_file.write(f'NMSE only with mu_out: {NMSE_only_mun:.4f}\n')
if MODEL_TYPE == 'Trajectory':
    glob_file.write(f'Mean Variance Prior: {m_sigma_squared_prior:.4f}\n')
    glob_file.write(f'Std Variance Prior: {std_sigma_squared_prior:.4f}\n')
    glob_file.write(f'Mean Variance Encoder: {m_sigma_squared_inf:.4f}\n')
    glob_file.write(f'Std Variance Encoder: {std_sigma_squared_inf:.4f}\n')
    if cov_type == 'Toeplitz':
        glob_file.write(f'Mean Alpha0: {m_alpha_0:4f}\n')
        glob_file.write(f'Std Alpha0: {std_alpha_0:4f}\n')
        glob_file.write(f'Number of Alphas (i>0) hitting their Bound: {n_bound_hits:4f}\n')
    if cov_type == 'DFT':
        glob_file.write(f'Mean Variance Decoder: {m_sigma_squared_out:4f}\n')
        glob_file.write(f'Std Variance Decoder: {std_sigma_squared_out:4f}\n')
