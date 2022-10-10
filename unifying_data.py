import h5py
import torch
import torch.nn as nn
import numpy as np

# def returning_H(number,property):
#     imag_data = h5py.File('../Simulations/trajectory_channel_prediction/data/H_imag_' + str(property) + str(number) + '.mat', 'r')
#     real_data = h5py.File('../Simulations/trajectory_channel_prediction/data/H_real_' + str(property) + str(number) + '.mat', 'r')
#     path_gain = h5py.File('../Simulations/trajectory_channel_prediction/data/path_gains_' + str(property) + str(number) + '.mat', 'r')
#     H_imag= np.array(imag_data['H_imag_' + str(property)])
#     H_real= np.array(real_data['H_real_' + str(property)])
#     path_gain = np.array(path_gain['path_gains_' + str(property)])
#     H_imag = torch.tensor(H_imag, dtype=torch.float)
#     H_real = torch.tensor(H_real, dtype=torch.float)
#     path_gain = torch.tensor(path_gain)
#     H = torch.stack((H_imag, H_real), dim=3)
#     H = H.permute(2, 3, 1, 0)
#     path_gain = path_gain.permute(1,0)
#     print(H.size())
#     print(path_gain.size())
#     return H,path_gain


imag_data = h5py.File('../Simulations/trajectory_channel_prediction/data/H_imag_Uma_mixed_IO_600_200_8snapshots_2ms.mat', 'r')
real_data = h5py.File('../Simulations/trajectory_channel_prediction/data/H_real_Uma_mixed_IO_600_200_8snapshots_2ms.mat', 'r')

#path_gain = h5py.File('../Simulations/trajectory_channel_prediction/data/path_gains120_1000.mat', 'r')
H_imag = np.array(imag_data['H_imag'])
H_real = np.array(real_data['H_real'])

#H_imag = H_imag[None,:,:]
#H_real = H_real[None,:,:]
#path_gain = np.array(path_gain['path_gains_total'])

H_imag = H_imag.transpose(2,1,0)
H_real = H_real.transpose(2,1,0)
#path_gain = path_gain.transpose(1,0)

H = H_real + 1j * H_imag

idx_shuffle = np.arange(120000)
np.random.shuffle(idx_shuffle)

H = H[idx_shuffle,:,:]

#print(type(H))
#path_gain = path_gain[idx_shuffle,:]

#H = H/np.sqrt(10**(0.1 * path_gain[:,-1][:,None,None]))

m_en = np.mean(np.sum(np.abs(H)**2,axis=1))

H = H/np.sqrt(m_en) * np.sqrt(32)

H_train = H[:100000,:,:]
H_val = H[100000:110000,:,:]
H_test = H[110000:,:,:]

#H_train = torch.zeros(10 * 4000,2,32,16)
#H_val = torch.zeros(10 * 500,2,32,16)
#H_test = torch.zeros(10 * 500,2,32,16)
#pg_train = torch.zeros(10 * 4000,16)
#pg_val = torch.zeros(10 * 500,16)
#pg_test = torch.zeros(10 * 500,16)

#for i in range(1,11):
#    H_test[(i-1) * 500 : i * 500,:,:,:],pg_test[(i-1) * 500 : i * 500,:] = returning_H(i,'test')
#    H_train[(i-1) * 4000 : i * 4000,:,:,:],pg_train[(i-1) * 4000 : i * 4000,:] = returning_H(i, 'train')
#    H_val[(i-1) * 500 : i * 500,:,:,:],pg_val[(i-1) * 500 : i * 500,:] = returning_H(i, 'val')
#H_test,pg_test = returning_H('500_100','test')
#H_train,pg_train = returning_H('500_100', 'train')
#H_val,pg_val = returning_H('500_100', 'val')


#H_train = np.array(H_train)
#H_val = np.array(H_val)
#H_test = np.array(H_test)
#pg_train = np.array(pg_train)
#pg_val = np.array(pg_val)
#pg_test = np.array(pg_test)

np.save('../Simulations/trajectory_channel_prediction/data/H_train_Uma_mixed_IO_600_200_8snapshots_2ms',H_train)
np.save('../Simulations/trajectory_channel_prediction/data/H_test_Uma_mixed_IO_600_200_8snapshots_2ms',H_test)
np.save('../Simulations/trajectory_channel_prediction/data/H_val_Uma_mixed_IO_600_200_8snapshots_2ms',H_val)
#np.save('../Simulations/trajectory_channel_prediction/data/pg_train',pg_train)
#np.save('../Simulations/trajectory_channel_prediction/data/pg_test',pg_test)
#np.save('../Simulations/trajectory_channel_prediction/data/pg_val',pg_val)