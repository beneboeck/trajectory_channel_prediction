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
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 50
G_EPOCHS = 9
LEARNING_RATE = 6e-5
FREE_BITS_LAMBDA = torch.tensor(1).to(device)
SNAPSHOTS = 16
DATASET_TYPE = 'my_Quadriga'
MODEL_TYPE = 'Trajectory' #Trajectory,Single,TraSingle
n_iterations = 75
n_permutations = 300
bs_mmd = 1000
normed = False
author = 'Bene'
SNR_db = 5

# CREATING FILES AND DIRECTORY
now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]

overall_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/'
dir_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_' + time
os.mkdir (dir_path)

if not(exists(overall_path + MODEL_TYPE + 'NAS_file.txt')):
    csvfile = open(overall_path + MODEL_TYPE + 'NAS_file.txt','w')
    csv_writer = csv.writer(csvfile)
    if MODEL_TYPE == 'Trajectory':
        csv_writer.writerow(['Time','LD', 'memory', 'rnn_bool', 'en_layer', 'en_width', 'pr_layer', 'pr_width', 'de_layer', 'de_width', 'cov_type', 'BN', 'prepro','Est','Pre','TPR','TPRinf'])
    if MODEL_TYPE == 'Single':
        csv_writer.writerow(['Time','LD_VAE', 'conv_layer', 'total_layer', 'out_channel', 'k_size', 'cov_type','prepro','Est'])
    csvfile.close()

glob_file = open(dir_path + '/glob_var_file.txt','w') # only the important results and the framework
log_file = open(dir_path + '/log_file.txt','w') # log_file which keeps track of the training and such stuff
glob_file.write('Date: ' +date +'\n')
glob_file.write('Time: ' + time + '\n\n')
glob_file.write(f'\nMODEL_TYPE: {MODEL_TYPE}\n\n')
glob_file.write(f'\AUTHER: {author}\n\n')
glob_file.write('BATCHSIZE: ' + str(BATCHSIZE) +'\n')
glob_file.write('G_EPOCHS: ' +str(G_EPOCHS) +'\n')
glob_file.write(f'Learning Rate: {LEARNING_RATE}\n')
glob_file.write(f'SNR_db: {SNR_db}\n')
glob_file.write(f'n_iterations: {n_iterations}\n')
glob_file.write(f'n_permutations: {n_permutations}\n\n')
log_file.write('Date: ' +date +'\n')
log_file.write('Time: ' + time + '\n')
log_file.write('global variables successfully defined\n\n')
print('global var successful')

# NETWORK ARCHITECTURE SEARCH
if MODEL_TYPE == 'Trajectory':
    LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type,BN,prepro,n_conv,cnn_bool = network_architecture_search()
    ## ACHTUNG, NACHAENDERUNG!!!!!!
    #LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type, BN, prepro = 14,1,False,2,8,2,6,4,8,'DFT',True,'DFT'
    LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type, BN, prepro,n_conv,cnn_bool = 32,10,True,3,4,3,3,4,6,'DFT',False,'None',2,True
    setup = [LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type,BN,prepro]
    print('Trajectory Setup')
    print(LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type,BN,prepro)
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
if MODEL_TYPE == 'Single':
    LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro = network_architecture_search_VAE()
    out_channel = 128
    setup = [LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro]
    print('Single Setup')
    print(LD_VAE,conv_layer,total_layer,out_channel,k_size,cov_type,prepro)
    glob_file.write(f'\nlatent Dim VAE: {LD_VAE}\n')
    glob_file.write(f'conv_layer: {conv_layer}\n')
    glob_file.write(f'total_layer: {total_layer}\n')
    glob_file.write(f'out_channel: {out_channel}\n')
    glob_file.write(f'k_size: {k_size}\n')
    glob_file.write(f'cov_type: {cov_type}\n')
    glob_file.write(f'prepro: {prepro}\n')
if MODEL_TYPE == 'TraSingle':
    LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro = network_architecture_search_TraVAE()
    setup = [LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro]
    print('Single Setup')
    print(LD_VAE,conv_layer,total_layer,out_channel,k_size,cov_type,prepro)
    glob_file.write(f'\nlatent Dim VAE: {LD_VAE}\n')
    glob_file.write(f'conv_layer: {conv_layer}\n')
    glob_file.write(f'total_layer: {total_layer}\n')
    glob_file.write(f'out_channel: {out_channel}\n')
    glob_file.write(f'k_size: {k_size}\n')
    glob_file.write(f'cov_type: {cov_type}\n')
    glob_file.write(f'prepro: {prepro}\n')

#LOADING AND PREPARING DATA + DEFINING THE MODEL

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
print('here')
# CREATING THE MODELS
if MODEL_TYPE == 'Trajectory':
    model = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,n_conv,cnn_bool,device).to(device)
if MODEL_TYPE == 'Single':
    model = mg.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,device).to(device)
    if author == 'Michael':
        model = mg.Michael_VAE_DFT(16,device).to(device)
if MODEL_TYPE == 'TraSingle':
    model = mg.my_tra_VAE(cov_type, LD_VAE, conv_layer, total_layer, out_channel, k_size, prepro,SNAPSHOTS, device).to(device)
    print('model generated')

risk_list,KL_list,RR_list,eval_risk,eval_NMSE, eval_NMSE_estimation, eval_TPR1,eval_TPR2 = tr.training_gen_NN(MODEL_TYPE,setup,LEARNING_RATE,cov_type, model, dataloader_train,dataloader_val, G_EPOCHS, FREE_BITS_LAMBDA,sig_n_val,device, log_file,dir_path,n_iterations, n_permutations, normed,bs_mmd, dataset_val, SNAPSHOTS)
model.eval()
save_risk(risk_list,RR_list,KL_list,dir_path,'Risks')

save_risk_single(eval_risk,dir_path,'Evaluation - ELBO')
save_risk_single(eval_NMSE,dir_path,'Evaluation - NMSE prediction')
save_risk_single(eval_NMSE_estimation,dir_path,'Evaluation - NMSE estimation')
save_risk_single(eval_TPR1,dir_path,'Evaluation - TPR1 prior')
save_risk_single(eval_TPR2,dir_path,'Evaluation - TPR2 - inference')

torch.save(model.state_dict(),dir_path + '/model_dict')
log_file.write('\nTESTING\n')
print('testing')
if MODEL_TYPE == 'Trajectory':
    NMSE_test = ev.channel_prediction(setup,model,dataloader_test,15,dir_path,device,'testing')
    TPR1, TPR2 = ev.computing_MMD(setup, model, n_iterations, n_permutations, normed, bs_mmd, dataset_test, SNAPSHOTS, dir_path, device)
    NMSE_val = ev.channel_prediction(setup, model, dataloader_val, 15, dir_path, device, 'testing')
    TPR1_val, TPR2_val = ev.computing_MMD(setup, model, n_iterations, n_permutations, normed, bs_mmd, dataset_val, SNAPSHOTS,dir_path, device)
    print(f'NMSE prediction test: {NMSE_test}')
    log_file.write(f'NMSE prediction test: {NMSE_test}\n')

NMSE_val_est = ev.channel_estimation(model,dataloader_val,sig_n_val,cov_type,dir_path,device)
NMSE_test_est = ev.channel_estimation(model,dataloader_test,sig_n_test,cov_type,dir_path,device)

csv_file = open(overall_path + MODEL_TYPE + 'NAS_file.txt','a')
csv_writer = csv.writer(csv_file)
if MODEL_TYPE == 'Trajectory':
    csv_writer.writerow([time,LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type, BN, prepro,NMSE_val_est,NMSE_val,TPR1_val,TPR2_val])
if MODEL_TYPE == 'Single':
    csv_writer.writerow([time,LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type,prepro,NMSE_val_est])

csv_file.close()


NMSE_LS,NMSE_sCov = ev.computing_LS_sample_covariance_estimator(dataset_val,sig_n_val)
print(f'LS,sCov estimation NMSE: {NMSE_LS:.4f},{NMSE_sCov:.4f}')
log_file.write(f'LS,sCov estimation NMSE: {NMSE_LS:.4f},{NMSE_sCov:.4f}\n')

glob_file.write('\nResults\n')
glob_file.write('EVALUATION SET\n')
glob_file.write(f'NMSE estimation: {eval_NMSE_estimation[-1]:.4f}\n')
glob_file.write(f'NMSE prediction: {eval_NMSE[-1]:.4f}\n')
glob_file.write(f'TPR - prior: {eval_TPR1[-1]:.4f}\n')
glob_file.write(f'TPR - inference: {eval_TPR2[-1]:.4f}\n')

glob_file.write('Test SET\n')
glob_file.write(f'NMSE estimation: {NMSE_test_est:.4f}\n')
if MODEL_TYPE == 'Trajectory':
    glob_file.write(f'NMSE prediction: {NMSE_test:.4f}\n')
    glob_file.write(f'TPR prior: {TPR1:.4f}\n')
    glob_file.write(f'TPR inf: {TPR2:.4f}\n')


