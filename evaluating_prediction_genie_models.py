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


path_DFT_Tra = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_17_00_tra/model_dict'
path_TN_Tra = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_09_50_tra/model_dict'
path_TD_Tra = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/models/time_17_08_tra/model_dict'

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
cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'DFT',14,False,1,2,6,2,8,4,8,True,'DFT'
model_DFT_Tra = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,device).to(device)
model_DFT_Tra.load_state_dict(torch.load(path_DFT_Tra,map_location=device))

cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'Toeplitz',10,False,9,4,6,3,8,5,12,False,'None'
model_TN_Tra = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,device).to(device)
model_TN_Tra.load_state_dict(torch.load(path_TN_Tra,map_location=device))

cov_type,LD,rnn_bool,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,BN,prepro = 'Toeplitz',14,True,6,2,3,3,4,5,8,False,'DFT'
model_TD_Tra = mg.HMVAE(cov_type,LD,rnn_bool,32,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,SNAPSHOTS,BN,prepro,device).to(device)
model_TD_Tra.load_state_dict(torch.load(path_TD_Tra,map_location=device))