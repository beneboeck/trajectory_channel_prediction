import numpy as np
import SCMMulti_MIMO as cg
import h5py
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *
import scipy
import time

path = '/home/ga42kab/lrz-nashome/Project_ToyExample/data/'

n_paths = 1
n_antennas = 32
n_snapshots = 21
n_samples = 40000

coords_list = np.load('coords_samples_npaths_1.npy')
#coords_list = None
sigma_grid_list = np.load('sigma_grid_samples_npaths_1.npy')
#sigma_grid_list = None
convex_sigmas_list = np.load('convex_sigmas_npaths_1.npy')
#convex_sigmas_list = None
print(coords_list.shape)

channel_generator = cg.SCMMulti(n_path=n_paths,coords_list = coords_list,sigma_grid_list = sigma_grid_list, convex_sigmas = convex_sigmas_list)
channel_generator.save_parameters(n_paths)

print(channel_generator.coords_list.shape)
print(channel_generator.sigma_grid_list.shape)
print(channel_generator.convex_sigmas)

t_0 = time.time()
h,t_BS,t_MS,gains,angles,interpolator_list = channel_generator.generate_channel(n_samples,n_snapshots,1,n_antennas,1)
t_1 = time.time()
print('exptected time')
print(f'{(t_1 - t_0)*1000 / (60*60)} h')
h_stacked = np.concatenate((np.real(h),np.imag(h)),axis=2)

h_observed = h_stacked[:,:16,:]
h_predicted = h_stacked[:,16:,:]
h_observed_train = h_observed[:32000,:,:]
h_observed_test = h_observed[32000:36000,:,:]
h_predicted_train = h_predicted[:32000,:,:]
h_predicted_test = h_predicted[32000:36000,:,:]
h_observed_val = h_observed[36000:,:,:]
h_predicted_val = h_predicted[36000:,:,:]

print(h_observed_train.shape)
print(h_observed_test.shape)
print(h_predicted_train.shape)
print(h_predicted_test.shape)

gains_observed = gains[:,:16,:]
gains_predicted = gains[:,16:,:]
gains_observed_train = gains_observed[:32000,:,:]
gains_observed_test = gains_observed[32000:36000,:,:]
gains_predicted_train = gains_predicted[:32000,:,:]
gains_predicted_test = gains_predicted[32000:36000,:,:]
gains_observed_val = gains_observed[36000:,:,:]
gains_predicted_val = gains_predicted[36000:,:,:]

angles_observed = angles[:,:16,:]
angles_predicted = angles[:,16:,:]
angles_observed_train = angles_observed[:32000,:,:]
angles_observed_test = angles_observed[32000:36000,:,:]
angles_predicted_train = angles_predicted[:32000,:,:]
angles_predicted_test = angles_predicted[32000:36000,:,:]
angles_observed_val = angles_observed[36000:,:,:]
angles_predicted_val = angles_predicted[36000:,:,:]

t_BS_observed = t_BS[:,:16,:]
t_BS_predicted = t_BS[:,16:,:]
t_BS_observed_train = t_BS_observed[:32000,:,:]
t_BS_observed_test = t_BS_observed[32000:36000,:,:]
t_BS_predicted_train = t_BS_predicted[:32000,:,:]
t_BS_predicted_test = t_BS_predicted[32000:36000,:,:]
t_BS_observed_val = t_BS_observed[36000:,:,:]
t_BS_predicted_val = t_BS_predicted[36000:,:,:]

print('gains')
print(gains_observed_test.shape)
print('angles')
print(angles_predicted_test.shape)
print('t_BS')
print(t_BS_predicted_test.shape)

np.save(path + 'y_train',h_predicted_train)
np.save(path + 'x_train',h_observed_train)
np.save(path + 'y_test',h_predicted_test)
np.save(path + 'x_test',h_observed_test)
np.save(path + 'y_val',h_predicted_val)
np.save(path + 'x_val',h_observed_val)

np.save(path + 'angles_y_train',angles_predicted_train)
np.save(path + 'angles_x_train',angles_observed_train)
np.save(path + 'angles_y_test',angles_predicted_test)
np.save(path + 'angles_x_test',angles_observed_test)
np.save(path + 'angles_y_val',angles_predicted_val)
np.save(path + 'angles_x_val',angles_observed_val)

np.save(path + 'gains_y_train',gains_predicted_train)
np.save(path + 'gains_x_train',gains_observed_train)
np.save(path + 'gains_y_test',gains_predicted_test)
np.save(path + 'gains_x_test',gains_observed_test)
np.save(path + 'gains_y_val',gains_predicted_val)
np.save(path + 'gains_x_val',gains_observed_val)

np.save(path + 't_BS_y_train',t_BS_predicted_train)
np.save(path + 't_BS_x_train',t_BS_observed_train)
np.save(path + 't_BS_y_test',t_BS_predicted_test)
np.save(path + 't_BS_x_test',t_BS_observed_test)
np.save(path + 't_BS_y_val',t_BS_predicted_val)
np.save(path + 't_BS_x_val',t_BS_observed_val)


# DoA_grid_test = np.linspace(-np.pi,np.pi,50)
# PL_grid_test = np.linspace(0,1,50)
# features_grid_test = []
# features_grid_test.append(PL_grid_test)
# features_grid_test.append(DoA_grid_test)
# coords_test = np.meshgrid(*features_grid_test,copy=False)
# coords_list_test = []
# for n in range(2):
#     coords_list_test.append(coords_test[n].reshape(-1,1))
# coords_list_test = np.squeeze(np.array(coords_list_test)).T
#
# y_grid = interpolator_list[0][0](coords_list_test)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_trisurf(coords_list_test[:,0], coords_list_test[:,1], np.squeeze(y_grid), cmap=cm.coolwarm,linewidth=0, antialiased=False)
# plt.show()
#
# for i in range(20):
#     plt.scatter(angles[i,:,0],gains[i,:,0])
# plt.xlim((-np.pi,np.pi))
# plt.ylim((0,1))
# plt.show()
#
# plt.imshow(np.real(h[0,:,:].T),cmap='hot')
# plt.show()
