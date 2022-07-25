import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datetime
from utils import *
import dataset as ds
import training as tr
import networks as mg
import evaluation as ev

GLOBAL_ARCHITECTURE = 'causal_kMemoryHMVAE'
# options: - 'kalmanVAE' - 'genericGlow' - 'markovVAE' - 'hiddenMarkovVanillaVAE' -
#             'markovVanillaVAE' - 'causal_kMemoryHMVAE' -'kMemoryHiddenMarkovVAE' - 'ApproxKMemoryHiddenMarkovVAE' -'kMemoryMarkovVAE' - 'WN_kMemoryHiddenMarkovVAE'
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

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

BATCHSIZE = 50
G_EPOCHS = 300
LEARNING_RATE = 3e-5
STANDARDIZE_METHOD = 'c_s' # 'pg_s','c_s' (c_s: conventional standardization, pg_s: path gain standardization)
FREE_BITS_LAMBDA = torch.tensor(1).to(device) # is negligible if free bits isn't used
SNAPSHOTS = 20 # 96 / 192 should be taken for all models expect the modelbased one
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

if (GLOBAL_ARCHITECTURE == 'causal_kMemoryHMVAE'):
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

print('test')
print(x_train.shape)
print(np.swapaxes(x_train,1,2)[:,:32,:].shape)
x_train_n = np.zeros((x_train.shape[0],2,32,x_train.shape[1]))
a = np.copy(np.swapaxes(x_train,1,2)[:,:,:32])
a = np.transpose(x_train,axes=(0,2,1))
a = a[:,:32,:]
print(a.shape)
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

if GLOBAL_ARCHITECTURE == 'kalmanVAE':
    if LOCAL_ARCHITECTURE == 'toeplitz':
        model = mg.KalmanVAE_toeplitz
    if LOCAL_ARCHITECTURE == 'toeplitz_same_dec':
        model = mg.KalmanVAE_toeplitz_same_dec
    if LOCAL_ARCHITECTURE == 'toeplitz_same_all':
        model = mg.KalmanVAE_toeplitz_same_all
    if LOCAL_ARCHITECTURE == 'unitary':
        model = mg.KalmanVAE_unitary
    if LOCAL_ARCHITECTURE == 'unitary_same_dec':
        model = mg.KalmanVAE_unitary_same_dec
    if LOCAL_ARCHITECTURE == 'unitary_same_all':
        model = mg.KalmanVAE_unitary_same_all
    if LOCAL_ARCHITECTURE == 'diagonal':
        model = mg.KalmanVAE_diagonal
    if LOCAL_ARCHITECTURE == 'diagonal_same_dec':
        model = mg.KalmanVAE_diagonal_same_dec
    if LOCAL_ARCHITECTURE == 'diagonal_same_all':
        model = mg.KalmanVAE_diagonal_same_all
    if LOCAL_ARCHITECTURE == 'diagonal_IF':
        model = mg.KalmanVAE_diagonal_IF

    iterations = DIM_VEC

if GLOBAL_ARCHITECTURE == 'causal_kMemoryHMVAE':
    if LOCAL_ARCHITECTURE == 'diagonal':
        model = mg.causal_kMemoryHMVAE_diagonal
    if LOCAL_ARCHITECTURE == 'toeplitz':
        model = mg.causal_kMemoryHMVAE_toeplitz

    iterations = DIM_VEC

if GLOBAL_ARCHITECTURE == 'ApproxKMemoryHiddenMarkovVAE':
    if LOCAL_ARCHITECTURE == 'diagonal':
        model = mg.ApproxKMemoryHiddenMarkovVAE_diagonal

    iterations = DIM_VEC

iteration = iterations[0]

if (GLOBAL_ARCHITECTURE == 'kalmanVAE'):
    model = model(iteration[0],iteration[1],iteration[2],iteration[3]).to(device)

if (GLOBAL_ARCHITECTURE == 'causal_kMemoryHMVAE'):
    model = model(iteration[0],iteration[1],iteration[2],iteration[3],iteration[4]).to(device)

if (GLOBAL_ARCHITECTURE == 'ApproxKMemoryHiddenMarkovVAE'):
    model = model(iteration[0],iteration[1],iteration[2],iteration[3],iteration[4]).to(device)

risk_list,KL_list,RR_list,eval_risk,eval_NMSE = tr.training_gen_NN(GLOBAL_ARCHITECTURE, iteration, LEARNING_RATE, model, dataloader,dataloader_val, G_EPOCHS, RISK_TYPE, FREE_BITS_LAMBDA,device, log_file,dir_path, dataset)

save_risk(risk_list,RR_list,KL_list,dir_path,'Risks')

save_risk_single(eval_risk,dir_path,'Evaluation - ELBO')
save_risk_single(eval_NMSE,dir_path,'Evaluation - NMSE')

torch.save(model.state_dict(),dir_path + '/model_dict')
log_file.write('\nTESTING\n')
print('testing')
NMSE_test = ev.channel_prediction(GLOBAL_ARCHITECTURE,model,dataloader_test,16,iteration,dir_path,device,'testing')
print(f'NMSE test: {NMSE_test}')
log_file.write(f'NMSE test: {NMSE_test}\n')