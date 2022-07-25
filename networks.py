import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import h5py
from scipy import linalg as la
import math
import time

class Reshape(nn.Module):
    def __init__(self,channels,x_size,y_size):
        super(Reshape, self).__init__()
        self.channels = channels
        self.x_size = x_size
        self.y_size = y_size
    def forward(self, x):
        return x.view(-1, self.channels, self.x_size, self.y_size)

class ReshapeDouble(nn.Module):
    def __init__(self,channel1,channel2,x_size,y_size):
        super(ReshapeDouble, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.x_size = x_size
        self.y_size = y_size
    def forward(self, x):
        return x.view(-1, self.channel1,self.channel2, self.x_size, self.y_size)

class kalmanEncUnit_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim):
        super(kalmanEncUnit_toeplitz,self).__init__()
        self.first_one = False
        self.z_dim = z_dim
        self.x_dim = x_dim

        input_size_1 = 1
        input_size_2 = 1
        for dim_1 in self.z_dim:
            input_size_1 = input_size_1 * dim_1
        for dim_2 in self.x_dim:
            input_size_2 = input_size_2 * dim_2

        self.input_size = input_size_1 + input_size_2


        if self.x_dim[2] != 1:
            self.x_prenet = nn.Sequential(
                nn.Conv2d(2,4,3,padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(4,track_running_stats=False),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(4,8,1),
                nn.ReLU(),
                nn.BatchNorm2d(8,track_running_stats=False),
            )

        if self.x_dim[2] == 1:
            self.x_prenet = nn.Sequential(
                nn.Conv2d(2,4,3,padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(4,track_running_stats=False),
                nn.Conv2d(4,8,1),
                nn.ReLU(),
                nn.BatchNorm2d(8,track_running_stats=False),
            )

        #
        self.z_prenet = nn.Sequential(
            nn.Linear(input_size_1,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,eps=0.02,track_running_stats=False),#
            nn.Linear(100,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,eps=0.02,track_running_stats=False),
        )

        if self.x_dim[2] != 1:
            self.net = nn.Sequential(
                nn.Linear(8 * self.x_dim[1]//2 * self.x_dim[2]//2 + 100,400),
                nn.ReLU(),
                nn.BatchNorm1d(400,track_running_stats=False),
                nn.Linear(400,400),
                nn.ReLU(),
                nn.BatchNorm1d(400,track_running_stats=False),
            )

        if self.x_dim[2] == 1:
            self.net = nn.Sequential(
                nn.Linear(8 * self.x_dim[1] * self.x_dim[2] + 100,400),
                nn.ReLU(),
                nn.BatchNorm1d(400,track_running_stats=False),
                nn.Linear(400,400),
                nn.ReLU(),
                nn.BatchNorm1d(400,track_running_stats=False),
            )

        self.mu_net = nn.Sequential(
            nn.Linear(400,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,input_size_1),
        )

        self.logvar_net = nn.Sequential(
            nn.Linear(400,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,input_size_1),
        )

    def forward(self,x,z):
        x = self.x_prenet(x)
        z = self.z_prenet(z)
        x = nn.Flatten()(x)
        #z = nn.Flatten()(z)
        cat_vec = torch.cat((x,z),dim=1)
        out = self.net(cat_vec)
        mu = self.mu_net(out)
        logvar = self.logvar_net(out)

        return mu,logvar

class kalmanDecUnit_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim,device):
        super(kalmanDecUnit_toeplitz,self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.device = device

        rand_matrix = torch.randn(32,32)
        self.B_mask = torch.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.B_mask = self.B_mask[None,None,:,:].to(self.device)

        self.C_mask = torch.tril(rand_matrix,diagonal=-1)
        self.C_mask[self.C_mask != 0] = 1
        self.C_mask = self.C_mask[None,None,:,:].to(self.device)

        input_size_1 = 1
        input_size_2 = 1
        for dim_1 in self.z_dim:
            input_size_1 *= dim_1
        for dim_2 in self.x_dim:
            input_size_2 *= dim_2

        self.net = nn.Sequential(
            nn.Linear(input_size_1,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
        )

        self.mu_out_net = nn.Sequential(
            nn.Linear(400,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            nn.Linear(400,int(input_size_2)),
            Reshape(self.x_dim[0],self.x_dim[1],self.x_dim[2]),
        )

        self.alpha_0_NN = nn.Sequential(
            nn.Linear(400,150),
            nn.ReLU(),
            nn.BatchNorm1d(150,track_running_stats=False),
            nn.Linear(150,1),)

        self.alpha_rest_NN = nn.Sequential(
            nn.Linear(400,300),
            nn.ReLU(),
            nn.BatchNorm1d(300,track_running_stats=False),
            nn.Linear(300,62),)



    def forward(self,z):
        z = nn.Flatten()(z)
        out = self.net(z)
        mu_out = self.mu_out_net(out)

        batchsize = out.size()[0]
        alpha_0 = self.alpha_0_NN(out)
        alpha_0 = torch.squeeze(Reshape(1, 1, 1)(alpha_0))
        alpha_0 = torch.exp(alpha_0)
        alpha_intermediate = alpha_0.clone()
        alpha_intermediate[alpha_0 > 500] = 500
        alpha_0 = alpha_intermediate.clone()
        alpha_rest = self.alpha_rest_NN(out)
        alpha_rest = Reshape(1, 1, 62)(alpha_rest)
        if batchsize != 1:
            alpha_0 = torch.squeeze(alpha_0)[:, None, None]
        if batchsize == 1:
            alpha_0 = torch.squeeze(alpha_0)[None, None, None]
        alpha_rest = torch.squeeze(alpha_rest)
        alpha_rest = alpha_rest[:,None,:]
        if batchsize == 1:
            alpha_rest = alpha_rest[None,None,:]

        alpha_rest = 0.022 * alpha_0 * nn.Tanh()(alpha_rest)
        alpha_rest = torch.complex(alpha_rest[:, :, :31], alpha_rest[:, :, 31:])
        Alpha = torch.cat((alpha_0, alpha_rest), dim=2)
        Alpha_prime = torch.cat((torch.zeros(batchsize, 1, 1).to(self.device), Alpha[:, :, 1:].flip(2)), dim=2)

        if batchsize > 100:
            del alpha_0, alpha_rest, alpha_intermediate
            torch.cuda.empty_cache()

        values = torch.cat((Alpha, Alpha[:, :, 1:].flip(2)), dim=2)
        i, j = torch.ones(32, 32).nonzero().T
        values = values[:, :, j - i].reshape(batchsize, 1, 32, 32)
        B = values * self.B_mask

        values_prime = torch.cat((Alpha_prime, Alpha_prime[:, :, 1:].flip(2)), dim=2)
        i, j = torch.ones(32, 32).nonzero().T
        values_prime2 = values_prime[:, :, j - i].reshape(batchsize, 1, 32, 32)
        C = torch.conj(values_prime2 * self.C_mask)
        return mu_out,B,C

class kalmanPriorUnit_toeplitz(nn.Module):
    def __init__(self,z_dim):
        super(kalmanPriorUnit_toeplitz,self).__init__()
        self.z_dim = z_dim

        self.input_size = 1
        for dim in z_dim:
            self.input_size *= dim

        self.net = nn.Sequential(
            nn.Linear(self.input_size,200),
            nn.ReLU(),
            #nn.BatchNorm1d(200), -> no BatchNorm for nets with a potential fixed input, since it makes the network unstable
            nn.BatchNorm1d(200,eps=0.02,track_running_stats=False),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,eps=0.02,track_running_stats=False),
        )

        self.mu_net = nn.Sequential(
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,self.input_size),
        )

        self.logpre_net = nn.Sequential(
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,self.input_size),

        )

    def forward(self,z):
        out = self.net(z)
        mu_out = self.mu_net(out)
        logpre_out = self.logpre_net(out)
        logpre_out[logpre_out > 4] = 4
        logpre_out[logpre_out < -4] = -4
        return mu_out,logpre_out

class KalmanVAE_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim,time_stamps_per_unit,device):
        super(KalmanVAE_toeplitz,self).__init__()

        self.device = device
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.z_size = 1
        self.x_size = 1
        self.time_stamps_per_unit = time_stamps_per_unit

        for dim in self.z_dim:
            self.z_size *= dim

        for dim in self.x_dim:
            self.x_size*= dim

        self.n_units = int(self.x_dim[2]/self.time_stamps_per_unit)
        self.x_dim_per_unit = [self.x_dim[0],self.x_dim[1],self.time_stamps_per_unit]

        self.encoder = nn.ModuleList([kalmanEncUnit_toeplitz(self.z_dim,self.x_dim_per_unit) for i in range(self.n_units)])
        self.decoder = nn.ModuleList([kalmanDecUnit_toeplitz(self.z_dim,self.x_dim_per_unit,self.device) for i in range(self.n_units)])
        self.prior_model = nn.ModuleList([kalmanPriorUnit_toeplitz(self.z_dim) for i in range(self.n_units)])

        self.encoder[0].first_one = True


    def reparameterize(self, log_var, mu):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std,eps

    def feed_prior(self,z):
        n_units = int(z.size()[2])
        batchsize = z.size()[0]
        z_init = torch.zeros(batchsize,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        mu_prior = torch.zeros(z.size()).to(self.device)
        logpre_prior = torch.zeros(z.size()).to(self.device)
        mu_prior[:,:,0],logpre_prior[:,:,0] = self.prior_model[0](z_init)
        for unit in range(1,n_units):
            z_input = z[:,:,unit-1].clone()
            mu_prior[:,:,unit],logpre_prior[:,:,unit] = self.prior_model[unit](z_input)
            del z_input

        #logpre_prior[logpre_prior > 6] = 6

        return mu_prior,logpre_prior

    def sample_from_prior(self,n_samples):
        z_init = torch.zeros(n_samples,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        z = torch.zeros(n_samples,self.z_dim[0],self.n_units).to(self.device)

        mu,logpre = self.prior_model[0](z_init)
        eps = torch.randn(n_samples,self.z_dim[0]).to(self.device)
        z_sample = mu + eps * 1/torch.sqrt(torch.exp(logpre)) # at the moment I am really implementing log_pre not log_var
        #z_sample = mu + eps * torch.exp(0.5 * logpre)
        z[:,:,0] = torch.squeeze(z_sample)

        for unit in range(1,self.n_units):
            mu,logpre = self.prior_model[unit](z[:,:,unit-1])
            eps = torch.randn(n_samples, self.z_dim[0]).to(self.device)
            z_sample = mu + eps * 1/torch.sqrt(torch.exp(logpre))
            #z_sample = mu + eps * torch.exp(0.5 * logpre)
            z[:,:,unit] = torch.squeeze(z_sample)

        return z

    def encode(self,x):
        batchsize = x.size()[0]
        z = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        z_init = torch.ones(batchsize,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        mu_inf = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        logvar_inf = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        eps = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        mu_z,logvar_z = self.encoder[0](x[:,:,:,0:self.time_stamps_per_unit],z_init)
        z_local,eps_local = self.reparameterize(logvar_z,mu_z)
        z[:,:,0] = z_local
        eps[:,:,0] = eps_local
        mu_inf[:,:,0] = mu_z
        logvar_inf[:,:,0] = logvar_z

        for unit in range(1,self.n_units):
            z_input = z[:,:,unit-1].clone()
            mu_z,logvar_z = self.encoder[unit](x[:,:,:,unit*self.time_stamps_per_unit:(unit+1)*self.time_stamps_per_unit],z_input)
            z_local,eps_local = self.reparameterize(logvar_z,mu_z)
            z[:,:,unit] = z_local
            eps[:,:,unit] = eps_local
            mu_inf[:,:,unit] = mu_z
            logvar_inf[:,:,unit] = logvar_z
            del z_local
            del eps_local
            del mu_z
            del logvar_z

        return z,eps,mu_inf,logvar_inf

    def decode(self,z):
        batchsize = z.size()[0]
        mu_out = torch.zeros([batchsize] + self.x_dim).to(self.device)
        B_out = torch.zeros(batchsize, self.x_dim[2],self.x_dim[1],self.x_dim[1],dtype=torch.cfloat).to(self.device)
        C_out = torch.zeros(batchsize, self.x_dim[2],self.x_dim[1],self.x_dim[1],dtype=torch.cfloat).to(self.device)

        for unit in range(self.n_units):
            z_input = z[:,:,unit].clone()
            mu_out_local, B_out_local,C_out_local = self.decoder[unit](z_input)
            #logpre_out_local[logpre_out_local > 9] = 9
            mu_out[:, :, :, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit] = mu_out_local
            B_out[:,unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit,:,:] = B_out_local
            C_out[:,unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit,:,:] = C_out_local
            del mu_out_local
            del B_out_local
            del C_out_local

        return mu_out,B_out,C_out


    def forward(self,x):
        z,eps,mu_inf,logvar_inf = self.encode(x)
        mu_out,B_out,C_out = self.decode(z)

        return mu_out,B_out,C_out,z,eps,mu_inf,logvar_inf



class kalmanEncUnit_diagonal(nn.Module):
    def __init__(self,z_dim,x_dim):
        super(kalmanEncUnit_diagonal,self).__init__()
        self.first_one = False
        self.z_dim = z_dim
        self.x_dim = x_dim

        input_size_1 = 1
        input_size_2 = 1
        for dim_1 in self.z_dim:
            input_size_1 = input_size_1 * dim_1
        for dim_2 in self.x_dim:
            input_size_2 = input_size_2 * dim_2

        self.input_size = input_size_1 + input_size_2


        if self.x_dim[2] != 1:
            self.x_prenet = nn.Sequential(
                nn.Conv2d(2,4,3,padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(4,track_running_stats=False),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(4,8,1),
                nn.ReLU(),
                nn.BatchNorm2d(8,track_running_stats=False),
            )

        if self.x_dim[2] == 1:
            self.x_prenet = nn.Sequential(
                nn.Conv2d(2,4,3,padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(4,track_running_stats=False),
                nn.Conv2d(4,8,1),
                nn.ReLU(),
                nn.BatchNorm2d(8,track_running_stats=False),
            )

        #
        self.z_prenet = nn.Sequential(
            nn.Linear(input_size_1,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,eps=0.02,track_running_stats=False),#
            nn.Linear(100,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,eps=0.02,track_running_stats=False),
        )

        if self.x_dim[2] != 1:
            self.net = nn.Sequential(
                nn.Linear(8 * self.x_dim[1]//2 * self.x_dim[2]//2 + 100,400),
                nn.ReLU(),
                nn.BatchNorm1d(400,track_running_stats=False),
                nn.Linear(400,400),
                nn.ReLU(),
                nn.BatchNorm1d(400,track_running_stats=False),
            )

        if self.x_dim[2] == 1:
            self.net = nn.Sequential(
                nn.Linear(8 * self.x_dim[1] * self.x_dim[2] + 100,400),
                nn.ReLU(),
                nn.BatchNorm1d(400,track_running_stats=False),
                nn.Linear(400,400),
                nn.ReLU(),
                nn.BatchNorm1d(400,track_running_stats=False),
            )

        self.mu_net = nn.Sequential(
            nn.Linear(400,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,input_size_1),
        )

        self.logvar_net = nn.Sequential(
            nn.Linear(400,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,input_size_1),
        )

    def forward(self,x,z):
        x = self.x_prenet(x)
        z = self.z_prenet(z)
        x = nn.Flatten()(x)
        #z = nn.Flatten()(z)
        cat_vec = torch.cat((x,z),dim=1)
        out = self.net(cat_vec)
        mu = self.mu_net(out)
        logvar = self.logvar_net(out)

        return mu,logvar

class kalmanDecUnit_diagonal(nn.Module):
    def __init__(self,z_dim,x_dim):
        super(kalmanDecUnit_diagonal,self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        input_size_1 = 1
        input_size_2 = 1
        for dim_1 in self.z_dim:
            input_size_1 *= dim_1
        for dim_2 in self.x_dim:
            input_size_2 *= dim_2

        self.net = nn.Sequential(
            nn.Linear(input_size_1,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
        )

        self.mu_out_net = nn.Sequential(
            nn.Linear(400,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            nn.Linear(400,int(input_size_2)),
            Reshape(self.x_dim[0],self.x_dim[1],self.x_dim[2]),
        )

        self.logpre_out_net = nn.Sequential(
            nn.Linear(400,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            nn.Linear(400,int(input_size_2/2)),
            Reshape(1,self.x_dim[1],self.x_dim[2])
        )

    def forward(self,z):
        z = nn.Flatten()(z)
        out = self.net(z)
        mu_out = self.mu_out_net(out)
        logpre_out = self.logpre_out_net(out)
        return mu_out,logpre_out

class kalmanPriorUnit_diagonal(nn.Module):
    def __init__(self,z_dim):
        super(kalmanPriorUnit_diagonal,self).__init__()
        self.z_dim = z_dim

        self.input_size = 1
        for dim in z_dim:
            self.input_size *= dim

        self.net = nn.Sequential(
            nn.Linear(self.input_size,200),
            nn.ReLU(),
            #nn.BatchNorm1d(200), -> no BatchNorm for nets with a potential fixed input, since it makes the network unstable
            nn.BatchNorm1d(200,eps=0.02,track_running_stats=False),
            nn.Linear(200,200),
            nn.ReLU(),
            #nn.BatchNorm1d(200),
            nn.BatchNorm1d(200, eps=0.02,track_running_stats=False)
        )

        self.mu_net = nn.Sequential(
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,self.input_size),
        )

        self.logpre_net = nn.Sequential(
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,self.input_size),

        )


    def forward(self,z):
        out = self.net(z)
        mu_out = self.mu_net(out)
        logpre_out = self.logpre_net(out)
        logpre_out[logpre_out > 4] = 4
        logpre_out[logpre_out < -4] = -4
        return mu_out,logpre_out

class KalmanVAE_diagonal(nn.Module):
    def __init__(self,z_dim,x_dim,time_stamps_per_unit,device):
        super(KalmanVAE_diagonal,self).__init__()

        self.device = device
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.z_size = 1
        self.x_size = 1
        self.time_stamps_per_unit = time_stamps_per_unit

        for dim in self.z_dim:
            self.z_size *= dim

        for dim in self.x_dim:
            self.x_size*= dim

        self.n_units = int(self.x_dim[2]/self.time_stamps_per_unit)
        self.x_dim_per_unit = [self.x_dim[0],self.x_dim[1],self.time_stamps_per_unit]

        self.encoder = nn.ModuleList([kalmanEncUnit_diagonal(self.z_dim,self.x_dim_per_unit) for i in range(self.n_units)])
        self.decoder = nn.ModuleList([kalmanDecUnit_diagonal(self.z_dim,self.x_dim_per_unit) for i in range(self.n_units)])
        self.prior_model = nn.ModuleList([kalmanPriorUnit_diagonal(self.z_dim) for i in range(self.n_units)])

        self.encoder[0].first_one = True


    def reparameterize(self, log_var, mu):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std,eps

    def feed_prior(self,z):
        n_units = int(z.size()[2])
        batchsize = z.size()[0]
        z_init = torch.zeros(batchsize,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        mu_prior = torch.zeros(z.size()).to(self.device)
        logpre_prior = torch.zeros(z.size()).to(self.device)
        mu_prior[:,:,0],logpre_prior[:,:,0] = self.prior_model[0](z_init)
        for unit in range(1,n_units):
            z_input = z[:,:,unit-1].clone()
            mu_prior[:,:,unit],logpre_prior[:,:,unit] = self.prior_model[unit](z_input)
            del z_input

        #logpre_prior[logpre_prior > 6] = 6

        return mu_prior,logpre_prior

    def sample_from_prior(self,n_samples):
        z_init = torch.zeros(n_samples,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        z = torch.zeros(n_samples,self.z_dim[0],self.n_units).to(self.device)

        mu,logpre = self.prior_model[0](z_init)
        eps = torch.randn(n_samples,self.z_dim[0]).to(self.device)
        z_sample = mu + eps * 1/torch.sqrt(torch.exp(logpre)) # at the moment I am really implementing log_pre not log_var
        #z_sample = mu + eps * torch.exp(0.5 * logpre)
        z[:,:,0] = torch.squeeze(z_sample)

        for unit in range(1,self.n_units):
            mu,logpre = self.prior_model[unit](z[:,:,unit-1])
            eps = torch.randn(n_samples, self.z_dim[0]).to(self.device)
            z_sample = mu + eps * 1/torch.sqrt(torch.exp(logpre))
            #z_sample = mu + eps * torch.exp(0.5 * logpre)
            z[:,:,unit] = torch.squeeze(z_sample)


        return z

    def encode(self,x):
        batchsize = x.size()[0]
        z = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        z_init = torch.ones(batchsize,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        mu_inf = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        logvar_inf = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        eps = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        mu_z,logvar_z = self.encoder[0](x[:,:,:,0:self.time_stamps_per_unit],z_init)
        z_local,eps_local = self.reparameterize(logvar_z,mu_z)
        z[:,:,0] = z_local
        eps[:,:,0] = eps_local
        mu_inf[:,:,0] = mu_z
        logvar_inf[:,:,0] = logvar_z

        for unit in range(1,self.n_units):
            z_input = z[:,:,unit-1].clone()
            mu_z,logvar_z = self.encoder[unit](x[:,:,:,unit*self.time_stamps_per_unit:(unit+1)*self.time_stamps_per_unit],z_input)
            z_local,eps_local = self.reparameterize(logvar_z,mu_z)
            z[:,:,unit] = z_local
            eps[:,:,unit] = eps_local
            mu_inf[:,:,unit] = mu_z
            logvar_inf[:,:,unit] = logvar_z
            del z_local
            del eps_local
            del mu_z
            del logvar_z

        return z,eps,mu_inf,logvar_inf

    def decode(self,z):
        batchsize = z.size()[0]
        mu_out = torch.zeros([batchsize] + self.x_dim).to(self.device)
        logpre_out = torch.zeros(batchsize, 1, self.x_dim[1], self.x_dim[2]).to(self.device)

        for unit in range(self.n_units):
            z_input = z[:,:,unit].clone()
            mu_out_local, logpre_out_local = self.decoder[unit](z_input)
            #logpre_out_local[logpre_out_local > 9] = 9
            mu_out[:, :, :, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit], logpre_out[:, :,:,unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit] = mu_out_local, logpre_out_local
            del mu_out_local
            del logpre_out_local

        return mu_out,logpre_out

    def forward(self,x):
        z,eps,mu_inf,logvar_inf = self.encode(x)
        mu_out,logpre_out = self.decode(z)

        return mu_out,logpre_out,z,eps,mu_inf,logvar_inf


class kMemoryHiddenMarkovEnc_Unit_diagonal(nn.Module):
    def __init__(self,z_dim,x_dim):
        super(kMemoryHiddenMarkovEnc_Unit_diagonal,self).__init__()
        self.first_one = False
        self.z_dim = z_dim
        self.x_dim = x_dim

        input_size_1 = 1
        input_size_2 = 1
        for dim_1 in self.z_dim:
            input_size_1 = input_size_1 * dim_1
        for dim_2 in self.x_dim:
            input_size_2 = input_size_2 * dim_2

        self.input_size = input_size_1 + input_size_2


        self.x_prenet = nn.Sequential(
            nn.Conv2d(2,5,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2,ceil_mode=True),
            nn.BatchNorm2d(5,track_running_stats=False),
            nn.Conv2d(5,10,1),
            nn.ReLU(),
            nn.BatchNorm2d(10,track_running_stats=False),
        )

        #
        self.z_prenet = nn.Sequential(
            nn.Linear(input_size_1,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,eps=0.02,track_running_stats=False),#
            nn.Linear(100,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,eps=0.02,track_running_stats=False),
        )

        self.net = nn.Sequential(
            nn.Linear(10 * math.ceil(self.x_dim[2]/2) * self.x_dim[1]//2 + 100,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            nn.Linear(400,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            )

        self.mu_net = nn.Sequential(
            nn.Linear(400,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,input_size_1),
        )

        self.logvar_net = nn.Sequential(
            nn.Linear(400,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,input_size_1),
        )

    def forward(self,x,z):
        x = self.x_prenet(x)
        z = self.z_prenet(z)
        x = nn.Flatten()(x)
        #z = nn.Flatten()(z)
        cat_vec = torch.cat((x,z),dim=1)
        out = self.net(cat_vec)
        mu = self.mu_net(out)
        logvar = self.logvar_net(out)

        return mu,logvar

class kMemoryHiddenMarkovDec_Unit_diagonal(nn.Module):
    def __init__(self,z_dim,x_dim):
        super(kMemoryHiddenMarkovDec_Unit_diagonal,self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        input_size_1 = 1
        input_size_2 = 1
        for dim_1 in self.z_dim:
            input_size_1 *= dim_1
        for dim_2 in self.x_dim:
            input_size_2 *= dim_2

        self.net = nn.Sequential(
            nn.Linear(input_size_1,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
        )

        self.mu_out_net = nn.Sequential(
            nn.Linear(400,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            nn.Linear(400,int(input_size_2)),
            Reshape(self.x_dim[0],self.x_dim[1],self.x_dim[2]),
        )

        self.logpre_out_net = nn.Sequential(
            nn.Linear(400,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            nn.Linear(400,int(input_size_2/2)),
            Reshape(1,self.x_dim[1],self.x_dim[2])
        )

    def forward(self,z):
        z = nn.Flatten()(z)
        out = self.net(z)
        mu_out = self.mu_out_net(out)
        logpre_out = self.logpre_out_net(out)
        return mu_out,logpre_out

class kMemoryHiddenMarkovPrior_Unit_diagonal(nn.Module):
    def __init__(self,z_dim):
        super(kMemoryHiddenMarkovPrior_Unit_diagonal,self).__init__()
        self.z_dim = z_dim

        self.input_size = 1
        for dim in z_dim:
            self.input_size *= dim

        self.net = nn.Sequential(
            nn.Linear(self.input_size,200),
            nn.ReLU(),
            #nn.BatchNorm1d(200), -> no BatchNorm for nets with a potential fixed input, since it makes the network unstable
            nn.BatchNorm1d(200,eps=0.02,track_running_stats=False),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,eps=0.02,track_running_stats=False),
        )

        self.mu_net = nn.Sequential(
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,self.input_size),
        )

        self.logpre_net = nn.Sequential(
            nn.Linear(200,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,self.input_size),

        )

    def forward(self,z):
        out = self.net(z)
        mu_out = self.mu_net(out)
        logpre_out = self.logpre_net(out)
        logpre_out[logpre_out > 4] = 4
        logpre_out[logpre_out < -4] = -4
        return mu_out,logpre_out

class causal_kMemoryHMVAE_diagonal(nn.Module):
    def __init__(self,z_dim,x_dim,time_stamps_per_unit,memory,device):
        super(causal_kMemoryHMVAE_diagonal,self).__init__()

        self.device = device
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.z_size = 1
        self.x_size = 1
        self.time_stamps_per_unit = time_stamps_per_unit
        self.memory = memory

        for dim in self.z_dim:
            self.z_size *= dim

        for dim in self.x_dim:
            self.x_size*= dim

        self.n_units = int(self.x_dim[2]/self.time_stamps_per_unit)
        self.x_dim_per_unit = [self.x_dim[0],self.x_dim[1],self.time_stamps_per_unit]
        self.x_dim_encoder = [self.x_dim[0],self.x_dim[1],self.time_stamps_per_unit * (self.memory+1)]
        self.z_dim_decoder = [self.z_dim[0],1 + self.memory]

        self.encoder = nn.ModuleList([kMemoryHiddenMarkovEnc_Unit_diagonal(self.z_dim,self.x_dim_encoder) for i in range(self.n_units)]) ### x_dim richtig?
        self.decoder = nn.ModuleList([kMemoryHiddenMarkovDec_Unit_diagonal(self.z_dim_decoder,self.x_dim_per_unit) for i in range(self.n_units)]) ### x_dim richtig?
        self.prior_model = nn.ModuleList([kMemoryHiddenMarkovPrior_Unit_diagonal(self.z_dim) for i in range(self.n_units)])

        self.encoder[0].first_one = True


    def reparameterize(self, log_var, mu):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std,eps

    def feed_prior(self,z):
        n_units = int(z.size()[2])
        batchsize = z.size()[0]
        z_init = torch.zeros(batchsize,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        mu_prior = torch.zeros(z.size()).to(self.device)
        logpre_prior = torch.zeros(z.size()).to(self.device)
        mu_prior[:,:,0],logpre_prior[:,:,0] = self.prior_model[0](z_init)
        for unit in range(1,n_units):
            z_input = z[:,:,unit-1].clone()
            mu_prior[:,:,unit],logpre_prior[:,:,unit] = self.prior_model[unit](z_input)
            del z_input

        #logpre_prior[logpre_prior > 6] = 6

        return mu_prior,logpre_prior

    def sample_from_prior(self,n_samples):
        z_init = torch.zeros(n_samples,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        z = torch.zeros(n_samples,self.z_dim[0],self.n_units).to(self.device)

        mu,logpre = self.prior_model[0](z_init)
        eps = torch.randn(n_samples,self.z_dim[0]).to(self.device)
        z_sample = mu + eps * 1/torch.sqrt(torch.exp(logpre)) # at the moment I am really implementing log_pre not log_var
        #z_sample = mu + eps * torch.exp(0.5 * logpre)
        z[:,:,0] = torch.squeeze(z_sample)

        for unit in range(1,self.n_units):
            mu,logpre = self.prior_model[unit](z[:,:,unit-1])
            eps = torch.randn(n_samples, self.z_dim[0]).to(self.device)
            z_sample = mu + eps * 1/torch.sqrt(torch.exp(logpre))
            #z_sample = mu + eps * torch.exp(0.5 * logpre)
            z[:,:,unit] = torch.squeeze(z_sample)


        return z

    def encode(self,x):
        batchsize = x.size()[0]
        z = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        z_init = torch.ones(batchsize,self.z_dim[0]).to(self.device) # zeros instead of ones in the spirit of Glow
        x_start = torch.ones(batchsize,self.x_dim[0],self.x_dim[1],self.memory*self.time_stamps_per_unit).to(self.device)
        mu_inf = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        logvar_inf = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
        eps = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)

        x_input = torch.cat((x_start,x[:,:,:,0][:,:,:,None]),dim=2)
        mu_z, logvar_z = self.encoder[0](x_input, z_init)
        z_local, eps_local = self.reparameterize(logvar_z, mu_z)
        z[:, :, 0] = z_local
        eps[:, :, 0] = eps_local
        mu_inf[:, :, 0] = mu_z
        logvar_inf[:, :, 0] = logvar_z

        for i in range(1,self.memory):
            x_input = torch.cat((x_start[:,:,:,self.memory-i], x[:, :, :, :i+1]), dim=2)
            z_input = z[:,:,i-1].clone()
            mu_z, logvar_z = self.encoder[i](x_input,z_input)
            # logpre_out_local[logpre_out_local > 9] = 9
            z_local,eps_local = self.reparameterize(logvar_z,mu_z)
            z[:,:,i] = z_local
            eps[:,:,i] = eps_local
            mu_inf[:,:,i] = mu_z
            logvar_inf[:,:,i] = logvar_z

        for unit in range(self.memory,self.n_units):
            z_input = z[:,:,unit-1].clone()
            x_input = x[:,:,:,unit-self.memory:unit+1]
            mu_z,logvar_z = self.encoder[unit](x_input,z_input)
            z_local,eps_local = self.reparameterize(logvar_z,mu_z)
            z[:,:,unit] = z_local
            eps[:,:,unit] = eps_local
            mu_inf[:,:,unit] = mu_z
            logvar_inf[:,:,unit] = logvar_z


        return z,eps,mu_inf,logvar_inf

    def decode(self,z):
        batchsize = z.size()[0]
        mu_out = torch.zeros([batchsize] + self.x_dim).to(self.device)
        z_init = torch.ones(batchsize, self.z_dim[0],self.memory).to(self.device)
        logpre_out = torch.zeros(batchsize, 1, self.x_dim[1], self.x_dim[2]).to(self.device)

        for i in range(self.memory):
            z_input = torch.cat((z_init[:, :, :self.memory - i], z[:, :, :i+1]), dim=2)
            mu_out_local, logpre_out_local = self.decoder[i](z_input)
            # logpre_out_local[logpre_out_local > 9] = 9
            mu_out[:, :, :, i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit], logpre_out[:, :,:,i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit] = mu_out_local, logpre_out_local

        for unit in range(self.memory,self.n_units):
            z_input = z[:,:,unit-self.memory:unit+1].clone()
            mu_out_local, logpre_out_local = self.decoder[unit](z_input)
            #logpre_out_local[logpre_out_local > 9] = 9
            mu_out[:, :, :, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit], logpre_out[:, :,:,unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit] = mu_out_local, logpre_out_local

        return mu_out,logpre_out


    def forward(self,x):
        z,eps,mu_inf,logvar_inf = self.encode(x)
        mu_out,logpre_out = self.decode(z)

        return mu_out,logpre_out,z,eps,mu_inf,logvar_inf