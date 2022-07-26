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
import evaluation as ev

now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
dir_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_' + time
os.mkdir (dir_path)
glob_var_file = open(dir_path + '/glob_var_file.txt','w')
log_file = open(dir_path + '/log_file.txt','w')
m_file = open(dir_path + '/m_file.txt','w')

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

BATCHSIZE = 50
G_EPOCHS = 1
LEARNING_RATE = 3e-5
FREE_BITS_LAMBDA = torch.tensor(1).to(device) # is negligible if free bits isn't used
SNAPSHOTS = 20 # 96 / 192 should be taken for all models expect the modelbased one
DATASET_TYPE = 'Quadriga'
VELOCITY = 2

LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type = network_architecture_search()

print('Setup')
print(LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type)

setup = [LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type]

glob_var_file.write('Date: ' +date +'\n')
glob_var_file.write('Time: ' + time + '\n')
glob_var_file.write(f'Latent Dim: {LD}\n')
glob_var_file.write(f'Memory: {memory}\n')
glob_var_file.write(f'RNN Bool: {rnn_bool}\n')
glob_var_file.write(f'En_Layer: {en_layer}\n')
glob_var_file.write(f'En_Width: {en_width}\n')
glob_var_file.write(f'Pr_Layer: {pr_layer}\n')
glob_var_file.write(f'Pr_Width: {pr_width}\n')
glob_var_file.write(f'De_Layer: {de_layer}\n')
glob_var_file.write(f'De_Width: {de_width}\n')
glob_var_file.write(f'Cov_Type: {cov_type}\n')
glob_var_file.write('BATCHSIZE: ' + str(BATCHSIZE) +'\n')
glob_var_file.write('G_EPOCHS: ' +str(G_EPOCHS) +'\n')
glob_var_file.write(f'VELOCITY: {VELOCITY}\n')

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

x_train_n = np.zeros((x_train.shape[0],2,32,x_train.shape[1]))
x_train_n[:,0,:,:] = np.transpose(x_train,axes=(0,2,1))[:,:32,:]
x_train_n[:,1,:,:] = np.transpose(x_train,axes=(0,2,1))[:,32:,:]

y_train_n = np.zeros((y_train.shape[0],2,32,y_train.shape[1]))
y_train_n[:,0,:,:] = np.transpose(y_train,axes=(0,2,1))[:,:32,:]
y_train_n[:,1,:,:] =np.transpose(y_train,axes=(0,2,1))[:,32:,:]

x_val_n = np.zeros((x_val.shape[0],2,32,x_val.shape[1]))
x_val_n[:,0,:,:] = np.transpose(x_val,axes=(0,2,1))[:,:32,:]
x_val_n[:,1,:,:] =  np.transpose(x_val,axes=(0,2,1))[:,32:,:]

y_val_n = np.zeros((y_val.shape[0],2,32,y_val.shape[1]))
y_val_n[:,0,:,:] = np.transpose(y_val,axes=(0,2,1))[:,:32,:]
y_val_n[:,1,:,:] =  np.transpose(y_val,axes=(0,2,1))[:,32:,:]

x_test_n = np.zeros((x_test.shape[0],2,32,x_test.shape[1]))
x_test_n[:,0,:,:] = np.transpose(x_test,axes=(0,2,1))[:,:32,:]
x_test_n[:,1,:,:] =  np.transpose(x_test,axes=(0,2,1))[:,32:,:]

y_test_n = np.zeros((y_test.shape[0],2,32,y_train.shape[1]))
y_test_n[:,0,:,:] = np.transpose(y_test,axes=(0,2,1))[:,:32,:]
y_test_n[:,1,:,:] = np.transpose(y_test,axes=(0,2,1))[:,32:,:]

x_train_n = x_train_n[label_train == VELOCITY]
y_train_n = y_train_n[label_train == VELOCITY]
y_train_n = y_train_n[:,:,:,1:]

x_val_n = x_val_n[label_val == VELOCITY]
y_val_n = y_val_n[label_val == VELOCITY]
y_val_n = y_val_n[:,:,:,1:]

x_test_n = x_test_n[label_test == VELOCITY]
y_test_n = y_test_n[label_test == VELOCITY]
y_test_n = y_test_n[:,:,:,1:]

data = np.concatenate((x_train_n,y_train_n),axis=3)
dataset = ds.dataset(data)
dataloader = DataLoader(dataset,batch_size=BATCHSIZE,shuffle=True)

data_val = np.concatenate((x_val_n,y_val_n),axis=3)
dataset_val = ds.dataset(data_val)
dataloader_val = DataLoader(dataset_val,batch_size=4 * BATCHSIZE,shuffle=True)

data_test = np.concatenate((x_test_n,y_test_n),axis=3)
dataset_test = ds.dataset(data_test)
dataloader_test = DataLoader(dataset_test,batch_size=4 * BATCHSIZE,shuffle=True)

model = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,device).to(device)

risk_list,KL_list,RR_list,eval_risk,eval_NMSE = tr.training_gen_NN(setup,LEARNING_RATE,cov_type, model, dataloader,dataloader_val, G_EPOCHS, FREE_BITS_LAMBDA,device, log_file,dir_path)

save_risk(risk_list,RR_list,KL_list,dir_path,'Risks')

save_risk_single(eval_risk,dir_path,'Evaluation - ELBO')
save_risk_single(eval_NMSE,dir_path,'Evaluation - NMSE')

torch.save(model.state_dict(),dir_path + '/model_dict')
log_file.write('\nTESTING\n')
print('testing')
NMSE_test = ev.channel_prediction(setup,model,dataloader_val,16,dir_path,device,'testing')
print(f'NMSE test: {NMSE_test}')
log_file.write(f'NMSE test: {NMSE_test}\n')