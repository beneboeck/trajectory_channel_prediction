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


class dataset(Dataset):
    def __init__(self,h):
        super().__init__()
        self.h = torch.tensor(h)

    def __len__(self):
        return self.h.size(0)

    def __getitem__(self,idx):
        return self.h[idx,:,:]