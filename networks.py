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
        x_input = torch.cat((x_start,x[:,:,:,0][:,:,:,None]),dim=3)
        mu_z, logvar_z = self.encoder[0](x_input, z_init)
        z_local, eps_local = self.reparameterize(logvar_z, mu_z)
        z[:, :, 0] = z_local
        eps[:, :, 0] = eps_local
        mu_inf[:, :, 0] = mu_z
        logvar_inf[:, :, 0] = logvar_z

        for i in range(1,self.memory):
            x_input = torch.cat((x_start[:,:,:,:self.memory-i], x[:, :, :, :i+1]), dim=3)
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


class kMemoryHiddenMarkovEnc_Unit_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim):
        super(kMemoryHiddenMarkovEnc_Unit_toeplitz,self).__init__()
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

class kMemoryHiddenMarkovDec_Unit_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim,device):
        super(kMemoryHiddenMarkovDec_Unit_toeplitz,self).__init__()
        self.device = device
        rand_matrix = torch.randn(32,32)
        self.B_mask = torch.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.B_mask = self.B_mask[None,None,:,:].to(self.device)

        self.C_mask = torch.tril(rand_matrix,diagonal=-1)
        self.C_mask[self.C_mask != 0] = 1
        self.C_mask = self.C_mask[None,None,:,:].to(self.device)

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

        self.alpha_0_NN = nn.Sequential(
            nn.Linear(400,150),
            nn.ReLU(),
            nn.BatchNorm1d(150,track_running_stats=False),
            nn.Linear(150,1),
        )

        self.alpha_rest_NN = nn.Sequential(
            nn.Linear(400,300),
            nn.ReLU(),
            nn.BatchNorm1d(300,track_running_stats=False),
            nn.Linear(300,62),
        )

    def forward(self, z):
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
        alpha_rest = alpha_rest[:, None, :]
        if batchsize == 1:
            alpha_rest = alpha_rest[None, None, :]

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
        return mu_out, B, C

class kMemoryHiddenMarkovPrior_Unit_toeplitz(nn.Module):
    def __init__(self,z_dim):
        super(kMemoryHiddenMarkovPrior_Unit_toeplitz,self).__init__()
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

class causal_kMemoryHMVAE_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim,time_stamps_per_unit,memory,device):
        super(causal_kMemoryHMVAE_toeplitz,self).__init__()

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

        self.encoder = nn.ModuleList([kMemoryHiddenMarkovEnc_Unit_toeplitz(self.z_dim,self.x_dim_encoder) for i in range(self.n_units)]) ### x_dim richtig?
        self.decoder = nn.ModuleList([kMemoryHiddenMarkovDec_Unit_toeplitz(self.z_dim_decoder,self.x_dim_per_unit,self.device) for i in range(self.n_units)]) ### x_dim richtig?
        self.prior_model = nn.ModuleList([kMemoryHiddenMarkovPrior_Unit_toeplitz(self.z_dim) for i in range(self.n_units)])

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
        x_input = torch.cat((x_start,x[:,:,:,0][:,:,:,None]),dim=3)
        mu_z, logvar_z = self.encoder[0](x_input, z_init)
        z_local, eps_local = self.reparameterize(logvar_z, mu_z)
        z[:, :, 0] = z_local
        eps[:, :, 0] = eps_local
        mu_inf[:, :, 0] = mu_z
        logvar_inf[:, :, 0] = logvar_z

        for i in range(1,self.memory):
            x_input = torch.cat((x_start[:,:,:,:self.memory-i], x[:, :, :, :i+1]), dim=3)
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
        B_out = torch.zeros(batchsize, self.x_dim[2], self.x_dim[1], self.x_dim[1], dtype=torch.cfloat).to(self.device)
        C_out = torch.zeros(batchsize, self.x_dim[2], self.x_dim[1], self.x_dim[1], dtype=torch.cfloat).to(self.device)

        for i in range(self.memory):
            z_input = torch.cat((z_init[:, :, :self.memory - i], z[:, :, :i+1]), dim=2)
            mu_out_local, B_out_local,C_out_local = self.decoder[i](z_input)
            # logpre_out_local[logpre_out_local > 9] = 9
            mu_out[:, :, :, i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit], B_out[:,i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit,:,:] = mu_out_local, B_out_local
            C_out[:, i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit,:,:] = C_out_local

        for unit in range(self.memory,self.n_units):
            z_input = z[:,:,unit-self.memory:unit+1].clone()
            mu_out_local, B_out_local,C_out_local = self.decoder[unit](z_input)
            #logpre_out_local[logpre_out_local > 9] = 9
            mu_out[:, :, :, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit] = mu_out_local
            B_out[:, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit,:,:] = B_out_local
            C_out[:, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit, :, :] = C_out_local

        return mu_out,B_out,C_out

    def forward(self,x):
        z,eps,mu_inf,logvar_inf = self.encode(x)
        mu_out,B_out,C_out = self.decode(z)

        return mu_out,B_out,C_out,z,eps,mu_inf,logvar_inf




class causal_kMemoryHMVAE_small_Enc_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim):
        super(causal_kMemoryHMVAE_small_Enc_toeplitz,self).__init__()
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
            nn.Conv2d(2,4,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2,ceil_mode=True),
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
        )

        self.net = nn.Sequential(
            nn.Linear(8 * math.ceil(self.x_dim[2]/2) * self.x_dim[1]//2 + 100,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            nn.Linear(400,200),
            nn.ReLU(),
            nn.BatchNorm1d(200,track_running_stats=False),
            nn.Linear(200,2 * input_size_1),
        )

    def forward(self,x,z):
        x = self.x_prenet(x)
        z = self.z_prenet(z)
        x = nn.Flatten()(x)
        #z = nn.Flatten()(z)
        cat_vec = torch.cat((x,z),dim=1)
        mu,logvar = self.net(cat_vec).chunk(2,dim=1)

        return mu,logvar

class causal_kMemoryHMVAE_small_Dec_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim,device):
        super(causal_kMemoryHMVAE_small_Dec_toeplitz,self).__init__()
        self.device = device
        rand_matrix = torch.randn(32,32)
        self.B_mask = torch.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.B_mask = self.B_mask[None,None,:,:].to(self.device)

        self.C_mask = torch.tril(rand_matrix,diagonal=-1)
        self.C_mask[self.C_mask != 0] = 1
        self.C_mask = self.C_mask[None,None,:,:].to(self.device)

        self.z_dim = z_dim
        self.x_dim = x_dim

        input_size_1 = 1
        input_size_2 = 1
        for dim_1 in self.z_dim:
            input_size_1 *= dim_1
        for dim_2 in self.x_dim:
            input_size_2 *= dim_2

        self.net = nn.Sequential(
            nn.Linear(input_size_1,300),
            nn.ReLU(),
            nn.BatchNorm1d(300,track_running_stats=False),
        )

        self.mu_out_net = nn.Sequential(
            nn.Linear(300,400),
            nn.ReLU(),
            nn.BatchNorm1d(400,track_running_stats=False),
            nn.Linear(400,int(input_size_2)),
            Reshape(self.x_dim[0],self.x_dim[1],self.x_dim[2]),
        )

        self.alpha_NN = nn.Sequential(
            nn.Linear(300,150),
            nn.ReLU(),
            nn.BatchNorm1d(150,track_running_stats=False),
            nn.Linear(150,63),
        )


    def forward(self, z):
        z = nn.Flatten()(z)
        out = self.net(z)
        mu_out = self.mu_out_net(out)

        batchsize = out.size()[0]
        alpha = self.alpha_NN(out)
        alpha_0 = alpha[:,0][:,None]
        alpha_rest = alpha[:,1:]
        alpha_0 = torch.squeeze(Reshape(1, 1, 1)(alpha_0))
        alpha_0 = torch.exp(alpha_0)
        alpha_intermediate = alpha_0.clone()
        alpha_intermediate[alpha_0 > 500] = 500
        alpha_0 = alpha_intermediate.clone()
        alpha_rest = Reshape(1, 1, 62)(alpha_rest)
        if batchsize != 1:
            alpha_0 = torch.squeeze(alpha_0)[:, None, None]
        if batchsize == 1:
            alpha_0 = torch.squeeze(alpha_0)[None, None, None]
        alpha_rest = torch.squeeze(alpha_rest)
        alpha_rest = alpha_rest[:, None, :]
        if batchsize == 1:
            alpha_rest = alpha_rest[None, None, :]

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
        return mu_out, B, C

class causal_kMemoryHMVAE_small_Prior_toeplitz(nn.Module):
    def __init__(self,z_dim):
        super(causal_kMemoryHMVAE_small_Prior_toeplitz,self).__init__()
        self.z_dim = z_dim

        self.input_size = 1
        for dim in z_dim:
            self.input_size *= dim

        self.net = nn.Sequential(
            nn.Linear(self.input_size,300),
            nn.ReLU(),
            #nn.BatchNorm1d(200), -> no BatchNorm for nets with a potential fixed input, since it makes the network unstable
            nn.BatchNorm1d(300,eps=0.02,track_running_stats=False),
            nn.Linear(300,100),
            nn.ReLU(),
            nn.BatchNorm1d(100,track_running_stats=False),
            nn.Linear(100,2 * self.input_size),)


    def forward(self,z):
        mu_out,logpre_out = self.net(z).chunk(2,dim=1)
        logpre_out = logpre_out.clone()
        logpre_out[logpre_out > 4] = 4
        logpre_out[logpre_out < -4] = -4
        return mu_out,logpre_out

class causal_kMemoryHMVAE_small_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim,time_stamps_per_unit,memory,device):
        super(causal_kMemoryHMVAE_small_toeplitz,self).__init__()

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

        self.encoder = nn.ModuleList([causal_kMemoryHMVAE_small_Enc_toeplitz(self.z_dim,self.x_dim_encoder) for i in range(self.n_units)]) ### x_dim richtig?
        self.decoder = nn.ModuleList([causal_kMemoryHMVAE_small_Dec_toeplitz(self.z_dim_decoder,self.x_dim_per_unit,self.device) for i in range(self.n_units)]) ### x_dim richtig?
        self.prior_model = nn.ModuleList([causal_kMemoryHMVAE_small_Prior_toeplitz(self.z_dim) for i in range(self.n_units)])

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
        x_input = torch.cat((x_start,x[:,:,:,0][:,:,:,None]),dim=3)
        mu_z, logvar_z = self.encoder[0](x_input, z_init)
        z_local, eps_local = self.reparameterize(logvar_z, mu_z)
        z[:, :, 0] = z_local
        eps[:, :, 0] = eps_local
        mu_inf[:, :, 0] = mu_z
        logvar_inf[:, :, 0] = logvar_z

        for i in range(1,self.memory):
            x_input = torch.cat((x_start[:,:,:,:self.memory-i], x[:, :, :, :i+1]), dim=3)
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
        B_out = torch.zeros(batchsize, self.x_dim[2], self.x_dim[1], self.x_dim[1], dtype=torch.cfloat).to(self.device)
        C_out = torch.zeros(batchsize, self.x_dim[2], self.x_dim[1], self.x_dim[1], dtype=torch.cfloat).to(self.device)

        for i in range(self.memory):
            z_input = torch.cat((z_init[:, :, :self.memory - i], z[:, :, :i+1]), dim=2)
            mu_out_local, B_out_local,C_out_local = self.decoder[i](z_input)
            # logpre_out_local[logpre_out_local > 9] = 9
            mu_out[:, :, :, i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit], B_out[:,i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit,:,:] = mu_out_local, B_out_local
            C_out[:, i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit,:,:] = C_out_local

        for unit in range(self.memory,self.n_units):
            z_input = z[:,:,unit-self.memory:unit+1].clone()
            mu_out_local, B_out_local,C_out_local = self.decoder[unit](z_input)
            #logpre_out_local[logpre_out_local > 9] = 9
            mu_out[:, :, :, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit] = mu_out_local
            B_out[:, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit,:,:] = B_out_local
            C_out[:, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit, :, :] = C_out_local

        return mu_out,B_out,C_out

    def forward(self,x):
        z,eps,mu_inf,logvar_inf = self.encode(x)
        mu_out,B_out,C_out = self.decode(z)

        return mu_out,B_out,C_out,z,eps,mu_inf,logvar_inf


class RNN_prior_block(nn.Module):
    def __init__(self,latent_dim,width1,width2):
        super().__init__()

        self.forget = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.Sigmoid(),
        )

        self.choice = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.Sigmoid(),
        )

        self.candidates = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.Tanh(),
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim,width1),
            nn.ReLU(),
            nn.Linear(width1,width2),
            nn.ReLU(),
            nn.Linear(width2,2 * latent_dim),
        )

    def forward(self,z,h):
        forget_state = h * self.forget(z)
        new_state = forget_state + (self.choice(z) * self.candidates(z))
        transformed_z = self.net(z)
        mu,logvar = (nn.Tanh()(new_state) * transformed_z).chunk(2,dim=1)

        return mu,logvar,h

class RNN_inf_block(nn.Module):
    def __init__(self,latent_dim,x_dim,width1,width2):
        super().__init__()

        self.forget = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.Sigmoid(),
        )

        self.choice = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.Sigmoid(),
        )

        self.candidates = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.Tanh(),
        )

        self.net = nn.Sequential(
            nn.Linear(latent_dim + x_dim,width1),
            nn.ReLU(),
            nn.Linear(width1,width2),
            nn.ReLU(),
            nn.Linear(width2,2 * latent_dim),
        )

    def forward(self,z,x,h):
        x = nn.Flatten()(x)
        forget_state = h * self.forget(z)
        new_state = forget_state + (self.choice(z) * self.candidates(z))
        z_in = torch.cat((x,z),dim=1)
        transformed_z = self.net(z_in)
        mu,logvar = (nn.Tanh()(new_state) * transformed_z).chunk(2,dim=1)

        return mu,logvar,h

class RNN_aided_HMVAE_toeplitz(nn.Module):
    def __init__(self,z_dim,x_dim,memory,device):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.z_size = 1
        self.x_size = 1
        self.memory = memory

        for dim in self.z_dim:
            self.z_size *= dim

        for dim in self.x_dim:
            self.x_size*= dim

        self.n_units = self.x_dim[2]
        self.x_dim_per_unit = [self.x_dim[0],self.x_dim[1],1]
        self.x_dim_encoder = [self.x_dim[0],self.x_dim[1],1]
        self.z_dim_decoder = [self.z_dim[0],1]

        self.encoder = nn.ModuleList([RNN_inf_block(self.z_size,self.x_size,300,200) for i in range(self.n_units)]) ### x_dim richtig?
        self.decoder = nn.ModuleList([kalmanDecUnit_toeplitz(self.z_dim_decoder,self.x_dim_per_unit,self.device) for i in range(self.n_units)]) ### x_dim richtig?
        self.prior_model = nn.ModuleList([RNN_prior_block(self.z_size,200,200) for i in range(self.n_units)])

    def reparameterize(self, log_var, mu):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std, eps

    def feed_prior(self, z):
            n_units = int(z.size(2))
            batchsize = z.size(0)
            z_init = torch.zeros(batchsize, self.z_dim[0]).to(self.device)  # zeros instead of ones in the spirit of Glow
            mu_prior = torch.zeros(z.size()).to(self.device)
            logpre_prior = torch.zeros(z.size()).to(self.device)
            hidden_state = torch.zeros(z.size()).to(self.device)
            mu_prior[:, :, 0], logpre_prior[:, :, 0],hidden_state[:,:,0] = self.prior_model[0](z_init,z_init)
            for unit in range(1, n_units):
                z_input = z[:, :, unit - 1].clone()
                h_input = hidden_state[:,:,unit-1].clone()
                mu_prior[:, :, unit], logpre_prior[:, :, unit],hidden_state[:,:,unit] = self.prior_model[unit](z_input,h_input)
                del z_input

            # logpre_prior[logpre_prior > 6] = 6

            return mu_prior, logpre_prior

    def sample_from_prior(self, n_samples):
            z_init = torch.zeros(n_samples, self.z_dim[0]).to(self.device)  # zeros instead of ones in the spirit of Glow
            z = torch.zeros(n_samples, self.z_dim[0], self.n_units).to(self.device)
            hidden_state = torch.zeros(n_samples,self.z_dim[0],self.n_units)
            mu, logpre,hidden_state[:,:,0] = self.prior_model[0](z_init,z_init)
            eps = torch.randn(n_samples, self.z_dim[0]).to(self.device)
            z_sample = mu + eps * 1 / torch.sqrt(torch.exp(logpre))  # at the moment I am really implementing log_pre not log_var
            # z_sample = mu + eps * torch.exp(0.5 * logpre)
            z[:, :, 0] = torch.squeeze(z_sample)

            for unit in range(1, self.n_units):
                mu, logpre, hidden_state[:,:,unit] = self.prior_model[unit](z[:, :, unit - 1],hidden_state[:,:,unit-1])
                eps = torch.randn(n_samples, self.z_dim[0]).to(self.device)
                z_sample = mu + eps * 1 / torch.sqrt(torch.exp(logpre))
                # z_sample = mu + eps * torch.exp(0.5 * logpre)
                z[:, :, unit] = torch.squeeze(z_sample)

            return z

    def encode(self, x):
            batchsize = x.size()[0]
            z = torch.zeros(batchsize, self.z_dim[0], self.n_units).to(self.device)
            hidden_state = torch.zeros(batchsize,self.z_dim[0],self.n_units).to(self.device)
            z_init = torch.ones(batchsize, self.z_dim[0]).to(self.device)  # zeros instead of ones in the spirit of Glow
            mu_inf = torch.zeros(batchsize, self.z_dim[0], self.n_units).to(self.device)
            logvar_inf = torch.zeros(batchsize, self.z_dim[0], self.n_units).to(self.device)
            eps = torch.zeros(batchsize, self.z_dim[0], self.n_units).to(self.device)
            mu_z, logvar_z,hidden_state[:,:,0] = self.encoder[0](z_init,x[:,:,:,0],z_init)
            z_local, eps_local = self.reparameterize(logvar_z, mu_z)
            z[:, :, 0] = z_local
            eps[:, :, 0] = eps_local
            mu_inf[:, :, 0] = mu_z
            logvar_inf[:, :, 0] = logvar_z

            for unit in range(1, self.n_units):
                z_input = z[:, :, unit - 1].clone()
                x_input = x[:, :, :, unit:unit + 1]
                mu_z, logvar_z,hidden_state[:,:,unit] = self.encoder[unit](z_input,x_input,hidden_state[:,:,unit-1].clone())
                z_local, eps_local = self.reparameterize(logvar_z, mu_z)
                z[:, :, unit] = z_local
                eps[:, :, unit] = eps_local
                mu_inf[:, :, unit] = mu_z
                logvar_inf[:, :, unit] = logvar_z
            return z, eps, mu_inf, logvar_inf

    def decode(self, z):
            batchsize = z.size()[0]
            mu_out = torch.zeros([batchsize] + self.x_dim).to(self.device)
            z_init = torch.ones(batchsize, self.z_dim[0], self.memory).to(self.device)
            B_out = torch.zeros(batchsize, self.x_dim[2], self.x_dim[1], self.x_dim[1], dtype=torch.cfloat).to(
                self.device)
            C_out = torch.zeros(batchsize, self.x_dim[2], self.x_dim[1], self.x_dim[1], dtype=torch.cfloat).to(
                self.device)

            for i in range(self.memory):
                z_input = torch.cat((z_init[:, :, :self.memory - i], z[:, :, :i + 1]), dim=2)
                mu_out_local, B_out_local, C_out_local = self.decoder[i](z_input)
                # logpre_out_local[logpre_out_local > 9] = 9
                mu_out[:, :, :, i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit], B_out[:,
                                                                                                    i * self.time_stamps_per_unit:(
                                                                                                                                              i + 1) * self.time_stamps_per_unit,
                                                                                                    :,
                                                                                                    :] = mu_out_local, B_out_local
                C_out[:, i * self.time_stamps_per_unit:(i + 1) * self.time_stamps_per_unit, :, :] = C_out_local

            for unit in range(self.memory, self.n_units):
                z_input = z[:, :, unit - self.memory:unit + 1].clone()
                mu_out_local, B_out_local, C_out_local = self.decoder[unit](z_input)
                # logpre_out_local[logpre_out_local > 9] = 9
                mu_out[:, :, :, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit] = mu_out_local
                B_out[:, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit, :, :] = B_out_local
                C_out[:, unit * self.time_stamps_per_unit:(unit + 1) * self.time_stamps_per_unit, :, :] = C_out_local

            return mu_out, B_out, C_out

    def forward(self, x):
            z, eps, mu_inf, logvar_inf = self.encode(x)
            mu_out, B_out, C_out = self.decode(z)

            return mu_out, B_out, C_out, z, eps, mu_inf, logvar_inf








class Prior(nn.Module):
    def __init__(self,ld,rnn_bool,pr_layer,pr_width):
        super().__init__()
        self.rnn_bool = rnn_bool
        self.ld = ld
        if rnn_bool == True:
            self.forget = nn.Sequential(
                nn.Linear(ld, ld),
                nn.Sigmoid(),)
            self.choice = nn.Sequential(
                nn.Linear(ld, ld),
                nn.Sigmoid(),)
            self.candidates = nn.Sequential(
                nn.Linear(ld, ld),
                nn.Tanh(),)

        self.net = []
        self.net.append(nn.Linear(ld,pr_width * ld))
        self.net.append(nn.ReLU())
        #self.net.append(nn.BatchNorm1d(pr_width * ld,track_running_stats=False))
        #self.net.append(nn.BatchNorm1d(pr_width * ld, eps=1e-3))
        for l in range(pr_layer-2):
            self.net.append(nn.Linear(pr_width * ld, pr_width * ld))
            self.net.append(nn.ReLU())
            #self.net.append(nn.BatchNorm1d(pr_width * ld,track_running_stats=False))
            #self.net.append(nn.BatchNorm1d(pr_width * ld, eps=1e-3))
        self.net.append(nn.Linear(pr_width * ld,2 * ld))
        self.hidden_to_out = nn.Linear(self.ld, 2 * self.ld)
        self.net = nn.Sequential(*self.net)

    def forward(self, z, h=None):
        if self.rnn_bool == True:
            forget_state = h * self.forget(z)
            new_state = forget_state + (self.choice(z) * self.candidates(z))
        transformed_z = self.net(z)
        if self.rnn_bool == True:
            mu, logpre = (nn.Tanh()(self.hidden_to_out(new_state)) * transformed_z).chunk(2, dim=1)
        else:
            mu, logpre = transformed_z.chunk(2, dim=1)
            new_state = torch.zeros(z.size())

        logpre = (2.3 - 1.1)/2 * nn.Tanh()(logpre) + (2.3 - 1.1)/2 + 1.1

        logpre2 = logpre.clone()
        if torch.sum(logpre[torch.abs(logpre) > 2.3]) != 0:
            print(torch.max(torch.abs(logpre)))
            print('logpre was regularized')
            print(torch.max(torch.abs(logpre)))
            logpre2[logpre > 4] = 4
        return mu, logpre2, new_state

class Encoder(nn.Module):
    def __init__(self,n_ant,ld,memory,rnn_bool,en_layer,en_width):
        super().__init__()
        self.rnn_bool = rnn_bool
        self.ld = ld
        if rnn_bool == True:
            self.forget = nn.Sequential(
                nn.Linear(ld, ld),
                nn.Sigmoid(),)
            self.choice = nn.Sequential(
                nn.Linear(ld, ld),
                nn.Sigmoid(),)
            self.candidates = nn.Sequential(
                nn.Linear(ld, ld),
                nn.Tanh(),)

        step = round((n_ant * 2 * (memory+1) - 2*ld)/2)

        self.x_prenet = nn.Sequential(
            nn.Linear(n_ant * 2 * (memory+1),int(n_ant * 2 * (memory+1) - step)),
            nn.ReLU(),
            #nn.BatchNorm1d(int(n_ant * 2 * (memory+1) - step),track_running_stats=False),
            #nn.BatchNorm1d(int(n_ant * 2 * (memory + 1) - step), eps=1e-3),
            nn.Linear(int(n_ant * 2 * (memory+1) - step),2*ld),)

        self.net = []
        self.net.append(nn.Linear(3 * ld,en_width * ld))
        self.net.append(nn.ReLU())
        #self.net.append(nn.BatchNorm1d(en_width * ld,track_running_stats=False))
        #self.net.append(nn.BatchNorm1d(en_width * ld, eps=1e-3))
        for l in range(en_layer-2):
            self.net.append(nn.Linear(en_width * ld, en_width * ld))
            self.net.append(nn.ReLU())
            #self.net.append(nn.BatchNorm1d(en_width * ld,track_running_stats=False))
            #self.net.append(nn.BatchNorm1d(en_width * ld, eps=1e-3))
        self.net.append(nn.Linear(en_width * ld,2 * ld))
        self.hidden_to_out = nn.Linear(self.ld,2*self.ld)

        self.net = nn.Sequential(*self.net)

    def forward(self,x,z,h):
        x = nn.Flatten()(x)
        if self.rnn_bool == True:
            forget_state = h * self.forget(z)
            new_state = forget_state + (self.choice(z) * self.candidates(z))
        transformed_x = self.x_prenet(x)
        transformed_z = self.net(torch.cat((z,transformed_x),dim=1))
        if self.rnn_bool == True:
            mu, logvar = (nn.Tanh()(self.hidden_to_out(new_state)) * transformed_z).chunk(2, dim=1)
        else:
            mu, logvar = transformed_z.chunk(2, dim=1)
            new_state = torch.zeros(z.size())

        logvar = (4.6 + 1.1) / 2 * nn.Tanh()(logvar) + (4.6 + 1.1) / 2 - 4.6
        logvar2 = logvar.clone()
        if torch.sum(logvar[torch.abs(logvar) > 5]) != 0:
            print(torch.max(torch.abs(logvar)))
            print('logvar was regularized')
            print(torch.max(torch.abs(logvar)))
            logvar2[logvar > 4] = 4
        return mu, logvar, new_state

class Decoder(nn.Module):
    def __init__(self,cov_type,ld,n_ant,memory,de_layer,de_width,device):
        super().__init__()
        self.cov_type = cov_type
        self.n_ant = n_ant
        self.device = device
        rand_matrix = torch.randn(32,32)
        self.B_mask = torch.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.B_mask = self.B_mask[None,None,:,:].to(self.device)

        self.C_mask = torch.tril(rand_matrix,diagonal=-1)
        self.C_mask[self.C_mask != 0] = 1
        self.C_mask = self.C_mask[None,None,:,:].to(self.device)

        if (cov_type == 'DFT') | (cov_type == 'diagonal'):
            output_dim = 3 * n_ant
        if cov_type == 'Toeplitz':
            output_dim = 2 * n_ant + 63
        step = round((de_width * ld * memory+1 - output_dim)/de_layer)

        self.net = []
        net_in_dim = de_width * ld * (memory+1)
        net_out_dim = int(de_width * ld * (memory+1) - step)
        self.net.append(nn.Linear(ld*(memory+1),net_in_dim))
        self.net.append(nn.ReLU())
        #self.net.append(nn.BatchNorm1d(net_in_dim,track_running_stats=False))
        #self.net.append(nn.BatchNorm1d(net_in_dim, eps=1e-3))
        for l in range(de_layer-2):
            self.net.append(nn.Linear(net_in_dim,net_out_dim))
            self.net.append(nn.ReLU())
            #self.net.append(nn.BatchNorm1d(net_out_dim,track_running_stats=False))
            #self.net.append(nn.BatchNorm1d(net_out_dim, eps=1e-3))
            net_in_dim = net_out_dim
            net_out_dim = int(net_out_dim - step)
        self.net.append(nn.Linear(net_in_dim,output_dim))
        self.net = nn.Sequential(*self.net)

    def forward(self,z):
        z = nn.Flatten()(z)
        out = self.net(z)
        if (self.cov_type == 'DFT') | (self.cov_type == 'diagonal'):
            mu_out,logpre_out = out[:,:2*self.n_ant],out[:,2*self.n_ant:]
            mu_out = Reshape(2,32,1)(mu_out)
            logpre_out = logpre_out[:,:,None]
            #logpre_out[logpre_out > 4] = 4
            logpre_out = (2.3 - 1.1) / 2 * nn.Tanh()(logpre_out) + (2.3 - 1.1) / 2 + 1.1
            if torch.sum(logpre_out[torch.abs(logpre_out) > 4]) != 0:
                print('logpre_out was regularized')
            if torch.sum(logpre_out == 0) > 0:
                print('logpre_out wirklich 0')
            return mu_out,logpre_out

        if self.cov_type == 'Toeplitz':
            mu_out,alpha = out[:,:2*self.n_ant],out[:,2*self.n_ant:]
            mu_out = Reshape(2, 32, 1)(mu_out)
            batchsize = out.size(0)
            alpha_0 = alpha[:, 0][:, None]
            alpha_rest = alpha[:, 1:]
            alpha_0 = torch.squeeze(Reshape(1, 1, 1)(alpha_0))
            alpha_0 = torch.exp(alpha_0)
            alpha_intermediate = alpha_0.clone()
            if torch.sum(alpha_intermediate[alpha_0 > 100]) > 0:
                print('alpha regularized')
            alpha_intermediate[alpha_0 > 100] = 100
            alpha_0 = alpha_intermediate.clone()
            alpha_rest = Reshape(1, 1, 62)(alpha_rest)
            if batchsize != 1:
                alpha_0 = torch.squeeze(alpha_0)[:, None, None]
            if batchsize == 1:
                alpha_0 = torch.squeeze(alpha_0)[None, None, None]
            alpha_rest = torch.squeeze(alpha_rest)
            alpha_rest = alpha_rest[:, None, :]
            if batchsize == 1:
                alpha_rest = alpha_rest[None, None, :]

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

            return mu_out, B, C


class HMVAE(nn.Module):
    def __init__(self,cov_type,ld,rnn_bool,n_ant,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,snapshots,device):
        super().__init__()
        # attributes
        self.memory = memory
        self.n_ant = n_ant
        self.snapshots = snapshots
        self.ld = ld
        self.device = device
        self.cov_type = cov_type

        self.encoder = nn.ModuleList([Encoder(n_ant,ld,memory,rnn_bool,en_layer,en_width) for i in range(snapshots)])
        self.decoder = nn.ModuleList([Decoder(cov_type,ld,n_ant,memory,de_layer,de_width,self.device) for i in range(snapshots)])
        self.prior_model = nn.ModuleList([Prior(ld,rnn_bool,pr_layer,pr_width) for i in range(snapshots)])

    def reparameterize(self, log_var, mu):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std, eps

    def feed_prior(self, z):
            # z: [BS,latent_dim,snapshots]
            batchsize = z.size(0)
            z_init = torch.zeros(batchsize, self.ld).to(self.device)  # zeros instead of ones in the spirit of Glow
            mu_prior = torch.zeros(z.size()).to(self.device)
            logpre_prior = torch.zeros(z.size()).to(self.device)
            hidden_state = torch.zeros(z.size()).to(self.device)
            mu_prior[:, :, 0], logpre_prior[:, :, 0],hidden_state[:,:,0] = self.prior_model[0](z_init,z_init)
            if torch.sum(mu_prior[:,:,0] != mu_prior[:,:,0]) > 0:
                print('Nan in feed prior')
                raise ValueError
            for unit in range(1, self.snapshots):
                z_input = z[:, :, unit - 1].clone()
                h_input = hidden_state[:,:,unit-1].clone()
                mu_prior[:, :, unit], logpre_prior[:, :, unit],hidden_state[:,:,unit] = self.prior_model[unit](z_input, h_input)
                if torch.sum(mu_prior[:, :, unit] != mu_prior[:, :, unit]) > 0:
                    print('Nan in feed prior For loop')
                    raise ValueError
            # logpre_prior[logpre_prior > 6] = 6
            return mu_prior, logpre_prior

    def sample_from_prior(self, n_samples):
            z_init = torch.zeros(n_samples, self.ld).to(self.device)  # zeros instead of ones in the spirit of Glow
            z = torch.zeros(n_samples, self.ld, self.snapshots).to(self.device)
            hidden_state = torch.zeros(n_samples,self.ld,self.snapshots).to(self.device)
            mu, logpre,hidden_state[:,:,0] = self.prior_model[0](z_init,z_init)
            eps = torch.randn(n_samples, self.ld).to(self.device)
            z_sample = mu + eps * 1 / torch.sqrt(torch.exp(logpre))  # at the moment I am really implementing log_pre not log_var
            # z_sample = mu + eps * torch.exp(0.5 * logpre)
            z[:, :, 0] = torch.squeeze(z_sample)

            for unit in range(1, self.snapshots):
                mu, logpre, hidden_state[:,:,unit] = self.prior_model[unit](z[:, :, unit - 1],hidden_state[:,:,unit-1].clone())
                print('logpre sample from prior')
                print(torch.max(torch.abs(torch.exp(logpre))))
                if torch.sum(mu != mu) > 0:
                    print('Nan in sample from prior')
                    print(unit)
                    raise ValueError
                eps = torch.randn(n_samples, self.ld).to(self.device)
                z_sample = mu + eps * 1 / torch.sqrt(torch.exp(logpre))
                # z_sample = mu + eps * torch.exp(0.5 * logpre)
                z[:, :, unit] = torch.squeeze(z_sample)

            return z

    def encode(self, x):

        batchsize = x.size(0)
        z = torch.zeros(batchsize, self.ld, self.snapshots).to(self.device)
        hidden_state = torch.zeros(batchsize, self.ld, self.snapshots).to(self.device)
        z_init = torch.ones(batchsize, self.ld).to(self.device)  # zeros instead of ones in the spirit of Glow
        if self.memory > 0:
            x_start = torch.ones(batchsize, 2, 32, self.memory).to(self.device)
        mu_inf = torch.zeros(batchsize, self.ld, self.snapshots).to(self.device)
        logvar_inf = torch.zeros(batchsize, self.ld, self.snapshots).to(self.device)
        eps = torch.zeros(batchsize, self.ld, self.snapshots).to(self.device)

        if self.memory > 0:
            x_input = torch.cat((x_start, x[:, :, :, 0][:, :, :, None]), dim=3)
        else:
            x_input = x[:,:,:,:1]
        mu_z, logvar_z,hidden_state[:,:,0] = self.encoder[0](x_input,z_init, z_init)
        z_local, eps_local = self.reparameterize(logvar_z, mu_z)
        z[:, :, 0] = z_local
        eps[:, :, 0] = eps_local
        mu_inf[:, :, 0] = mu_z
        logvar_inf[:, :, 0] = logvar_z

        for i in range(1, self.memory):
            x_input = torch.cat((x_start[:, :, :, :self.memory - i], x[:, :, :, :i + 1]), dim=3)
            z_input = z[:, :, i - 1].clone()
            mu_z, logvar_z,hidden_state[:,:,i] = self.encoder[i](x_input, z_input,hidden_state[:,:,i-1].clone())
            if torch.sum(mu_z != mu_z) > 0:
                print('Nan in encode')
                raise ValueError
            # logpre_out_local[logpre_out_local > 9] = 9
            z_local, eps_local = self.reparameterize(logvar_z, mu_z)
            z[:, :, i] = z_local
            eps[:, :, i] = eps_local
            mu_inf[:, :, i] = mu_z
            logvar_inf[:, :, i] = logvar_z

        for unit in range(self.memory, self.snapshots):
            z_input = z[:, :, unit - 1].clone()
            x_input = x[:, :, :, unit - self.memory:unit + 1]
            mu_z, logvar_z,hidden_state[:,:,unit] = self.encoder[unit](x_input, z_input,hidden_state[:,:,unit-1].clone())
            if torch.sum(mu_z != mu_z) > 0:
                print('Nan in encode')
                raise ValueError
            z_local, eps_local = self.reparameterize(logvar_z, mu_z)
            z[:, :, unit] = z_local
            eps[:, :, unit] = eps_local
            mu_inf[:, :, unit] = mu_z
            logvar_inf[:, :, unit] = logvar_z
        return z, eps, mu_inf, logvar_inf


    def decode(self, z):
            batchsize = z.size(0)
            mu_out = torch.zeros(batchsize,2,self.n_ant,self.snapshots).to(self.device)
            z_init = torch.ones(batchsize,self.ld, self.memory).to(self.device)
            if self.cov_type == 'Toeplitz':
                B_out = torch.zeros(batchsize, self.snapshots, self.n_ant, self.n_ant, dtype=torch.cfloat).to(self.device)
                C_out = torch.zeros(batchsize, self.snapshots, self.n_ant, self.n_ant, dtype=torch.cfloat).to(self.device)
            else:
                logpre_out = torch.zeros(batchsize,self.n_ant,self.snapshots).to(self.device)

            for i in range(self.memory):
                z_input = torch.cat((z_init[:, :, :self.memory - i], z[:, :, :i + 1]), dim=2)
                if self.cov_type == 'Toeplitz':
                    mu_out_local, B_out_local, C_out_local = self.decoder[i](z_input)
                    mu_out[:, :, :, i:(i + 1)], B_out[:, i:(i + 1), :, :], C_out[:, i:(i + 1), :,:] = mu_out_local, B_out_local, C_out_local
                    if torch.sum(mu_out != mu_out) > 0:
                        print('Nan in decode')
                        raise ValueError
                else:
                    mu_out_local, logpre_local = self.decoder[i](z_input)
                    mu_out[:, :, :, i:(i + 1)],logpre_out[:,:,i:i+1] = mu_out_local, logpre_local
                # logpre_out_local[logpre_out_local > 9] = 9

            for unit in range(self.memory, self.snapshots):
                z_input = z[:, :, unit - self.memory:unit + 1].clone()
                if self.cov_type == 'Toeplitz':
                    mu_out_local, B_out_local, C_out_local = self.decoder[unit](z_input)
                    mu_out[:, :, :, unit :unit + 1] = mu_out_local
                    B_out[:, unit:unit + 1, :, :] = B_out_local
                    C_out[:, unit:unit + 1, :, :] = C_out_local
                else:
                    mu_out_local, logpre_local = self.decoder[unit](z_input)
                    mu_out[:, :, :, unit :unit + 1] = mu_out_local
                    logpre_out[:,:,unit:unit + 1] = logpre_local
                    # logpre_out_local[logpre_out_local > 9] = 9
                if torch.sum(mu_out_local != mu_out_local) > 0:
                    print('Nan in decode')
                    raise ValueError
            if self.cov_type == 'Toeplitz':
                return mu_out,B_out,C_out
            else:
                return mu_out,logpre_out


    def forward(self, x):
            z, eps, mu_inf, logvar_inf = self.encode(x)
            out = self.decode(z)

            return out, z, eps, mu_inf, logvar_inf


