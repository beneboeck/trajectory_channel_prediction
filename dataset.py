import numpy as np
import SCMMulti_MIMO as cg
import h5py
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *
import scipy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import networks as n

#H_test,H_test_dft,n_H_test, n_H_test_dft

class dataset(Dataset):
    def __init__(self,h,h_dft,y,y_dft):
        super().__init__()
        self.h = torch.tensor(h).float()
        self.y = torch.tensor(y).float()
        self.h_dft = torch.tensor(h_dft).float()
        self.y_dft = torch.tensor(y_dft).float()


    def __len__(self):
        return self.h.size(0)

    def __getitem__(self,idx):
        return self.h[idx,:,:,:],self.y[idx,:,:,:],self.h_dft[idx,:,:,:],self.y_dft[idx,:,:,:]