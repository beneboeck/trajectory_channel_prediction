import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import h5py
from scipy import linalg as la
import math
from utils import *
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

class Prior(nn.Module):
    def __init__(self,ld,rnn_bool,pr_layer,pr_width,BN):
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
        if BN:
            self.net.append(nn.BatchNorm1d(pr_width * ld, eps=1e-4))
        for l in range(pr_layer-2):
            self.net.append(nn.Linear(pr_width * ld, pr_width * ld))
            self.net.append(nn.ReLU())
            if BN:
                self.net.append(nn.BatchNorm1d(pr_width * ld, eps=1e-4))
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
        #logpre = (2.3 - 1.1) / 2 * nn.Tanh()(logpre) + (2.3 - 1.1) / 2 + 2.3
        logpre = (15 + 3.5)/2 * nn.Tanh()(logpre) + (15 + 3.5)/2 - 3.5
        logpre2 = logpre.clone()
        return mu, logpre2, new_state

class Encoder(nn.Module):
    def __init__(self,n_ant,ld,memory,rnn_bool,en_layer,en_width,BN,prepro,cov_type,device):
        super().__init__()
        self.rnn_bool = rnn_bool
        self.ld = ld
        self.prepro = prepro
        self.cov_type = cov_type
        self.device = device
        self.F = torch.zeros((n_ant, n_ant), dtype=torch.cfloat).to(self.device)
        for m in range(n_ant):
            for n in range(n_ant):
                self.F[m, n] = 1 / torch.sqrt(torch.tensor(n_ant)) * torch.exp(torch.tensor(1j * 2 * math.pi * (m * n) / n_ant))

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

        if BN:
            self.x_prenet = nn.Sequential(
                nn.Linear(n_ant * 2 * (memory+1),int(n_ant * 2 * (memory+1) - step)),
                nn.ReLU(),
                nn.BatchNorm1d(int(n_ant * 2 * (memory + 1) - step), eps=1e-4),
                nn.Linear(int(n_ant * 2 * (memory+1) - step),2*ld),)
        else:
            self.x_prenet = nn.Sequential(
                nn.Linear(n_ant * 2 * (memory+1),int(n_ant * 2 * (memory+1) - step)),
                nn.ReLU(),
                nn.Linear(int(n_ant * 2 * (memory+1) - step),2*ld),)

        self.net = []
        self.net.append(nn.Linear(3 * ld,en_width * ld))
        self.net.append(nn.ReLU())
        if BN:
            self.net.append(nn.BatchNorm1d(en_width * ld, eps=1e-4))
        for l in range(en_layer-2):
            self.net.append(nn.Linear(en_width * ld, en_width * ld))
            self.net.append(nn.ReLU())
            if BN:
                self.net.append(nn.BatchNorm1d(en_width * ld, eps=1e-4))
        self.net.append(nn.Linear(en_width * ld,2 * ld))
        self.hidden_to_out = nn.Linear(self.ld,2*self.ld)

        self.net = nn.Sequential(*self.net)

    def forward(self,x,z,h):
        if (self.prepro == 'DFT') & (self.cov_type == 'Toeplitz'):
            x_new = torch.zeros((x.size())).to(self.device)
            x = x[:, 0, :, :] + 1j * x[:, 1, :, :]
            transformed_set = torch.einsum('mn,knl -> kml', self.F, x)
            x_new[:, 0, :, :] = torch.real(transformed_set)
            x_new[:, 1, :, :] = torch.imag(transformed_set)
            x = nn.Flatten()(x_new)
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

        #logvar = (4.6 + 1.1) / 2 * nn.Tanh()(logvar) + (4.6 + 1.1) / 2 - 4.6
        logvar = (15 + 3.5) / 2 * nn.Tanh()(logvar) + (15 + 3.5) / 2 - 15
        return mu, logvar, new_state

class Decoder(nn.Module):
    def __init__(self,cov_type,ld,n_ant,memory,de_layer,de_width,BN,device):
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
        if BN:
            self.net.append(nn.BatchNorm1d(net_in_dim, eps=1e-4))
        for l in range(de_layer-2):
            self.net.append(nn.Linear(net_in_dim,net_out_dim))
            self.net.append(nn.ReLU())
            if BN:
                self.net.append(nn.BatchNorm1d(net_out_dim, eps=1e-4))
            net_in_dim = net_out_dim
            net_out_dim = int(net_out_dim - step)
        self.net.append(nn.Linear(net_in_dim,output_dim))
        self.net = nn.Sequential(*self.net)

    def forward(self,z):
        z = nn.Flatten()(z)
        out = self.net(z)
        if (self.cov_type == 'DFT') | (self.cov_type == 'diagonal'):
            mu_out,logpre_out = out[:,:2*self.n_ant],out[:,2*self.n_ant:]
            logpre_out = (0.5 + 11) / 2 * nn.Tanh()(logpre_out) + (0.5 + 11) / 2 - 0.5
            mu_out = Reshape(2,32,1)(mu_out)
            logpre_out = logpre_out[:,:,None]
            #logpre_out[logpre_out > 4] = 4
            #logpre_out = (2.3 - 1.1) / 2 * nn.Tanh()(logpre_out) + (2.3 - 1.1) / 2 + 1.1
            return mu_out,logpre_out

        if self.cov_type == 'Toeplitz':
            mu_out,alpha = out[:,:2*self.n_ant],out[:,2*self.n_ant:]
            mu_out = Reshape(2, 32, 1)(mu_out)
            batchsize = out.size(0)
            alpha_0 = alpha[:, 0][:, None]
            alpha_rest = alpha[:, 1:]
            if torch.sum(alpha_0[alpha_0 > 9996]) > 0:
                print('alpha regularized')
            alpha_0 = (10 + 2)/2 * nn.Tanh()(alpha_0) - 2 + (10 + 2)/2
            alpha_0 = torch.squeeze(Reshape(1, 1, 1)(alpha_0))
            alpha_0 = torch.exp(alpha_0)
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
    def __init__(self,cov_type,ld,rnn_bool,n_ant,memory,pr_layer,pr_width,en_layer,en_width,de_layer,de_width,snapshots,BN,prepro,device):
        super().__init__()
        # attributes
        self.memory = memory
        self.n_ant = n_ant
        self.snapshots = snapshots
        self.ld = ld
        self.device = device
        self.cov_type = cov_type
        self.BN = BN

        self.encoder = nn.ModuleList([Encoder(n_ant,ld,memory,rnn_bool,en_layer,en_width,BN,prepro,cov_type,self.device) for i in range(snapshots)])
        self.decoder = nn.ModuleList([Decoder(cov_type,ld,n_ant,memory,de_layer,de_width,BN,self.device) for i in range(snapshots)])
        self.prior_model = nn.ModuleList([Prior(ld,rnn_bool,pr_layer,pr_width,BN) for i in range(snapshots)])

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

    def predicting(self,sample,knowledge):

        # input: 1. complete sample # BS, 2, N_ANT, SNAPSHOTS, 2. KNOWLEDGE (number samples, which are assumed to be known)
        # method: take the mean of the first K encoder units in the latent space, input the latest ones into the prior network and complete the chains by the produced means of the prior network,
        #         then input the latent realizations into the decoder and take the means of the predicted snapshots as estimations
        # output: 1. remaining (S-K) channel vectors # BS, 2, N_ANT, SNAPSHOTS - KNOWLEDGE , 2. ground truth (S-K) channel vectors # BS, 2, N_ANT, SNAPSHOTS - KNOWLEDGE

        batchsize = sample.size(0)
        z = torch.zeros(batchsize, self.ld, self.snapshots).to(self.device)
        hidden_state_inf = torch.zeros(batchsize, self.ld, knowledge).to(self.device)
        hidden_state_prior = torch.zeros(batchsize,self.ld, self.snapshots).to(self.device)
        z_init = torch.ones(batchsize, self.ld).to(self.device)  # zeros instead of ones in the spirit of Glow
        if self.memory > 0:
            x_start = torch.ones(batchsize, 2, 32, self.memory).to(self.device)
            x_input = torch.cat((x_start, sample[:, :, :, :1]), dim=3)
        else:
            x_input = sample[:, :, :, :1]
        mu_z, logvar_z, hidden_state_inf[:, :, 0] = self.encoder[0](x_input, z_init, z_init)
        z[:, :, 0] = mu_z
        for i in range(1, self.memory):
            x_input = torch.cat((x_start[:, :, :, :self.memory - i], sample[:, :, :, :i + 1]), dim=3)
            z_input = z[:, :, i - 1].clone()
            mu_z, logvar_z, hidden_state_inf[:, :, i] = self.encoder[i](x_input, z_input,hidden_state_inf[:, :, i - 1].clone())
            # logpre_out_local[logpre_out_local > 9] = 9
            z[:, :, i] = mu_z
        for unit in range(self.memory, knowledge):
            z_input = z[:, :, unit - 1].clone()
            x_input = sample[:, :, :, unit - self.memory:unit + 1]
            mu_z, logvar_z, hidden_state_inf[:, :, unit] = self.encoder[unit](x_input, z_input,hidden_state_inf[:, :, unit - 1].clone())
            z[:, :, unit] = mu_z

        # prior
        z_input = z[:, :, knowledge - 1].clone()
        _, _, hidden_state_prior[:, :, 0] = self.prior_model[0](z_init, z_init)
        for unit in range(1,knowledge):
            z_input = z[:,:,unit-1].clone()
            _,_,hidden_state_prior[:,:,unit] = self.prior_model[unit](z_input,hidden_state_prior[:,:,unit-1].clone())
        for idx in range(knowledge, self.snapshots):
            z_local, _, hidden_state_prior[:, :, idx] = self.prior_model[idx](z_input, hidden_state_prior[:, :, idx - 1].clone())
            z[:, :, idx] = z_local
            z_input = z_local.clone()

        # prediction
        predicted_samples = torch.zeros(sample.size(0), 2, 32, self.snapshots - knowledge).to(self.device)
        for idx in range(knowledge, self.snapshots):
            x_local = self.decoder[idx](z[:, :, idx - self.memory:idx + 1])[0]
            predicted_samples[:, :, :, (idx - knowledge):(idx - knowledge + 1)] = x_local

        ground_truth_samples = sample[:, :, :, knowledge:]

        return predicted_samples,ground_truth_samples

    def estimating(self,sample,estimated_snapshot):

        # input: 1. complete sample # BS, 2, N_ANT, SNAPSHOTS, 2. estimated snapshot - number between 0 and SNAPSHOTS
        # method: input the sample into the encoder and forward the means in the latent space from snapshot to snapshot, then use the decoder of the desired snapshot to produce an output distribution
        # output: mean and covariance matrix of the snapshot of interest # BS,N_ANT complex valued, # BS,N_ANT,N_ANT complex valued

        batchsize = sample.size(0)
        z = torch.zeros(batchsize, self.ld, self.snapshots).to(self.device)
        hidden_state = torch.zeros(batchsize, self.ld, self.snapshots).to(self.device)
        z_init = torch.ones(batchsize,self.ld).to(self.device)  # zeros instead of ones in the spirit of Glow
        if self.memory > 0:
            x_start = torch.ones(batchsize, 2, 32, self.memory).to(self.device)
            x_input = torch.cat((x_start, sample[:, :, :, :1]), dim=3)
        else:
            x_input = sample[:, :, :, :1]
        mu_z, logvar_z, hidden_state[:, :, 0] = self.encoder[0](x_input, z_init, z_init)
        z[:, :, 0] = mu_z

        for i in range(1, self.memory):
            x_input = torch.cat((x_start[:, :, :, :self.memory - i], sample[:, :, :, :i + 1]), dim=3)
            z_input = z[:, :, i - 1].clone()
            mu_z, logvar_z, hidden_state[:, :, i] = self.encoder[i](x_input, z_input, hidden_state[:, :, i - 1].clone())
            z[:, :, i] = mu_z

        for unit in range(self.memory, self.snapshots):
            z_input = z[:, :, unit - 1].clone()
            x_input = sample[:, :, :, unit - self.memory:unit + 1]
            mu_z, logvar_z, hidden_state[:, :, unit] = self.encoder[unit](x_input, z_input,hidden_state[:, :, unit - 1].clone())
            z[:, :, unit] = mu_z

        #decoding
        if self.cov_type == 'Toeplitz':
            mu_out, B_out, C_out = self.decode(z)
        else:
            mu_out, logpre_out = self.decode(z)

        if self.cov_type == 'Toeplitz':
            alpha_0 = B_out[:, :, 0, 0]
            if len(alpha_0.size()) == 2:
                Gamma = 1 / alpha_0[:, :, None, None] * (torch.matmul(B_out, torch.conj(B_out).permute(0, 1, 3, 2)) - torch.matmul(C_out,torch.conj(C_out).permute(0,1,3,2)))
            if len(alpha_0.size()) == 1:
                Gamma = 1 / alpha_0[None, :, None, None] * (torch.matmul(B_out, torch.conj(B_out).permute(0, 1, 3, 2)) - torch.matmul(C_out,torch.conj(C_out).permute(0,1,3,2)))
            Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j
            L,U = torch.linalg.eigh(Gamma)
            Cov_out = U @ torch.diag_embed(1/L).cfloat() @ U.mH

        else:
            Cov_out = torch.diag_embed(1 / (torch.exp(logpre_out.permute(0, 2, 1)))).cfloat()

        mu_out = mu_out[:,0,:,:] + 1j * mu_out[:,1,:,:]

        mu_out_interest = mu_out[:,:,estimated_snapshot]
        Cov_out_interest = Cov_out[:,estimated_snapshot,:,:]

        return mu_out_interest,Cov_out_interest

    def forward(self, x):
        z, eps, mu_inf, logvar_inf = self.encode(x)
        out = self.decode(z)

        return out, z, eps, mu_inf, logvar_inf

class Michael_VAE_DFT(nn.Module):
    def __init__(self,LD,device):
        super().__init__()
        self.latent_dim = LD
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv1d(1,8,7,2,1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8,32,7,2,1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,128,7,2,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(5 * 128, self.latent_dim)
        self.fc_var = nn.Linear(5 * 128, self.latent_dim)


        self.decoder_input = nn.Linear(self.latent_dim, 5 * 128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128,32,7,2,1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 8, 7, 2, 1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 7, 2, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

        self.final_layer = nn.Linear(61, 96)

    def encode(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        mu, log_std = self.fc_mu(out), self.fc_var(out)
        return mu, log_std

    def reparameterize(self, log_var, mu):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self,z):
        out = self.decoder_input(z)
        bs = out.size(0)
        out = out.view(bs,128,-1)
        out = self.decoder(out)
        out = torch.squeeze(out)
        out = self.final_layer(out)
        mu_real,mu_imag,log_pre = out.chunk(3,dim=1)
        mu_out = torch.zeros(bs,2,32).to(self.device)
        mu_out[:,0,:] = mu_real
        mu_out[:,1,:] = mu_imag
        return mu_out,log_pre

    def estimating(self,x,estimated_snapshot):
        x = x[:,:,:,estimated_snapshot]
        x = nn.Flatten()(x)
        x = x[:,None,:]
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        mu_out,log_pre = self.decode(z)
        mu_out = mu_out[:, 0, :] + 1j * mu_out[:, 1, :]
        Cov_out = torch.diag_embed(1 / (torch.exp(log_pre))) + 0j
        return mu_out, Cov_out

    def forward(self, x):
        x = nn.Flatten()(x)
        x = x[:,None,:]
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        mu_out,log_pre = self.decode(z)
        Gamma = torch.diag_embed(torch.exp(log_pre)) + 0j
        return mu_out,Gamma, mu, log_var


class my_VAE(nn.Module):
    def __init__(self,cov_type,ld,conv_layer,total_layer,out_channels,k_size,prepro,device):
        super().__init__()
        rand_matrix = torch.randn(32, 32)
        self.device = device
        self.prepro = prepro
        self.F = torch.zeros((32, 32), dtype=torch.cfloat).to(self.device)
        for m in range(32):
            for n in range(32):
                self.F[m, n] = 1 / torch.sqrt(torch.tensor(32)) * torch.exp(torch.tensor(1j * 2 * math.pi * (m * n) / 32))
        self.B_mask = torch.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.B_mask = self.B_mask[None, :, :].to(self.device)

        self.C_mask = torch.tril(rand_matrix, diagonal=-1)
        self.C_mask[self.C_mask != 0] = 1
        self.C_mask = self.C_mask[None, :, :].to(self.device)
        self.latent_dim = ld
        self.conv_layer = conv_layer
        self.cov_type = cov_type
        self.total_layer = total_layer
        self.out_channels = out_channels
        self.k_size = k_size
        if conv_layer > 0:
            step = int(math.floor((out_channels - 2)/conv_layer))
        self.encoder = []
        in_channels = 2
        for i in range(conv_layer-1):
            self.encoder.append(nn.Conv1d(in_channels,in_channels + step,k_size,2,int((k_size-1)/2)))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm1d(in_channels + step))
            in_channels = in_channels + step
        if conv_layer > 0:
            self.encoder.append(nn.Conv1d(in_channels,out_channels,k_size,2,int((k_size-1)/2)))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm1d(out_channels))

        in_linear = 64
        if conv_layer > 0:
            self.encoder.append(nn.Flatten())
            for i in range(total_layer-conv_layer):
                self.encoder.append(nn.Linear(int(32/(2**conv_layer) * out_channels),int(32/(2**conv_layer) * out_channels)))
                self.encoder.append(nn.ReLU())
                self.encoder.append(nn.BatchNorm1d(int(32/(2**conv_layer) * out_channels)))

            self.fc_mu = nn.Linear(int(32 / (2 ** conv_layer) * out_channels), self.latent_dim)
            self.fc_var = nn.Linear(int(32 / (2 ** conv_layer) * out_channels), self.latent_dim)
        else:
            self.encoder.append(nn.Linear(int(in_linear),int(out_channels/4 * in_linear)))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm1d(int(out_channels/4 * in_linear)))
            for i in range(1,total_layer - conv_layer - 1):
                self.encoder.append(nn.Linear(int(out_channels / 4 * in_linear),int( out_channels / 4 * in_linear)))
                self.encoder.append(nn.ReLU())
                self.encoder.append(nn.BatchNorm1d(int(out_channels / 4 * in_linear)))

            self.fc_mu = nn.Linear(int(out_channels / 4 * in_linear), self.latent_dim)
            self.fc_var = nn.Linear(int(out_channels / 4 * in_linear), self.latent_dim)
        self.encoder = nn.Sequential(*self.encoder)
        dim_out = 0
        self.decoder_lin = []
        if conv_layer > 0:
            self.decoder_input = nn.Linear(self.latent_dim,int(32 / (2 ** conv_layer) * out_channels))
            for i in range(total_layer-conv_layer):
                self.decoder_lin.append(nn.Linear(int(32/(2**conv_layer) * out_channels),int(32/(2**conv_layer) * out_channels)))
                self.decoder_lin.append(nn.ReLU())
                self.decoder_lin.append(nn.BatchNorm1d(int(32/(2**conv_layer) * out_channels)))
            dim_out = int(32/(2**conv_layer) * out_channels)
        else:
            self.decoder_input = nn.Linear(self.latent_dim, int(out_channels / 4 * in_linear))
            for i in range(1,total_layer - conv_layer - 1):
                self.decoder_lin.append(nn.Linear(int(out_channels / 4 * in_linear), int(out_channels / 4 * in_linear)))
                self.decoder_lin.append(nn.ReLU())
                self.decoder_lin.append(nn.BatchNorm1d(int(out_channels / 4 * in_linear)))
            dim_out = int(out_channels / 4 * in_linear)

        self.decoder_lin = nn.Sequential(*self.decoder_lin)
        self.decoder = []
        if conv_layer > 0:
            dim_out = dim_out / out_channels
        for i in range(conv_layer - 1):
            self.decoder.append(nn.ConvTranspose1d(out_channels, out_channels - step, k_size, 2))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.BatchNorm1d(out_channels - step))
            out_channels = out_channels - step
            dim_out = (dim_out-1) * 2 + (k_size-1) + 1
        #Lout=(Lin−1)×stride−2×padding + dilation×(kernel_size−1) + output_padding + 1


        if conv_layer > 0:
            self.decoder.append(nn.ConvTranspose1d(out_channels, 2, k_size, 2))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.BatchNorm1d(2))
            dim_out = (dim_out - 1) * 2 + (k_size - 1) + 1

        self.decoder = nn.Sequential(*self.decoder)
        if cov_type == 'DFT':
            if self.conv_layer > 0:
                self.final_layer = nn.Linear(int(2 * dim_out), 96)
            else:
                self.final_layer = nn.Linear(int(dim_out), 96)
        if cov_type == 'Toeplitz':
            if self.conv_layer > 0:
                self.final_layer = nn.Linear(int(2 * dim_out),64 + 63)
            else:
                self.final_layer = nn.Linear(int(dim_out), 64 + 63)

    def encode(self, x):
        if (self.cov_type == 'Toeplitz') & (self.prepro == 'DFT'):
            x_new = torch.zeros((x.size())).to(self.device)
            x = x[:, 0, :] + 1j * x[:, 1, :]
            transformed_set = torch.einsum('mn,kn -> km', self.F, x)
            x_new[:, 0, :] = torch.real(transformed_set)
            x_new[:, 1, :] = torch.imag(transformed_set)
            x = x_new
        if self.conv_layer == 0:
            x = nn.Flatten()(x)
        out = self.encoder(x)
        out = nn.Flatten()(out)
        mu, log_var = self.fc_mu(out), self.fc_var(out)
        log_var = (15 + 2.5) / 2 * nn.Tanh()(log_var) + (15 + 2.5) / 2 - 15
        return mu, log_var

    def reparameterize(self, log_var, mu):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mu + eps * std

    def decode(self,z):
        batchsize = z.size(0)
        out = self.decoder_input(z)
        bs = out.size(0)
        out = self.decoder_lin(out)
        if self.conv_layer > 0:
            out = out.view(bs,self.out_channels,-1)
        out = self.decoder(out)
        out = nn.Flatten()(out)

        out = self.final_layer(out)
        if self.cov_type == 'DFT':
            mu_real,mu_imag,log_pre = out.chunk(3,dim=1)
            log_pre = (0.5 + 15) / 2 * nn.Tanh()(log_pre) + (0.5 + 15) / 2 - 0.5
            mu_out = torch.zeros(batchsize,2,32).to(self.device)
            mu_out[:,0,:] = mu_real
            mu_out[:,1,:] = mu_imag
            return mu_out, log_pre

        if self.cov_type == 'Toeplitz':
            mu_real, mu_imag,alpha = out[:,:32],out[:,32:64],out[:,64:]
            alpha_0 = alpha[:, 0][:, None]
            alpha_rest = alpha[:, 1:]
            alpha_0 = torch.exp(alpha_0)
            alpha_intermediate = alpha_0.clone()
            if torch.sum(alpha_intermediate[alpha_0 > 5000]) > 0:
                print('alpha regularized')
            alpha_intermediate[alpha_0 > 5000] = 5000
            alpha_0 = alpha_intermediate.clone()
            alpha_rest = torch.squeeze(alpha_rest)
            alpha_rest = 0.022 * alpha_0 * nn.Tanh()(alpha_rest)
            alpha_rest = torch.complex(alpha_rest[:, :31], alpha_rest[:, 31:])
            Alpha = torch.cat((alpha_0, alpha_rest), dim=1)
            Alpha_prime = torch.cat((torch.zeros(batchsize, 1).to(self.device), Alpha[:, 1:].flip(1)), dim=1)
            values = torch.cat((Alpha, Alpha[:, 1:].flip(1)), dim=1)
            i, j = torch.ones(32, 32).nonzero().T
            values = values[:, j - i].reshape(batchsize, 32, 32)
            B = values * self.B_mask

            values_prime = torch.cat((Alpha_prime, Alpha_prime[:, 1:].flip(1)), dim=1)
            i, j = torch.ones(32, 32).nonzero().T
            values_prime2 = values_prime[:, j - i].reshape(batchsize, 32, 32)
            C = torch.conj(values_prime2 * self.C_mask)
            mu_out = torch.zeros(batchsize,2,32).to(self.device)
            mu_out[:,0,:] = mu_real
            mu_out[:,1,:] = mu_imag
            return mu_out,B,C

    def estimating(self,x,estimated_snapshot):
        x = x[:,:,:,estimated_snapshot]
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        if self.cov_type == 'DFT':
            mu_out,log_pre = self.decode(z)
            mu_out = mu_out[:,0,:] + 1j * mu_out[:,1,:]
            Cov_out = torch.diag_embed(1 / (torch.exp(log_pre))) + 0j
        if self.cov_type == 'Toeplitz':
            mu_out,B,C = self.decode(z)
            mu_out = mu_out[:, 0, :] + 1j * mu_out[:, 1, :]
            alpha_0 = B[:, 0, 0]
            Gamma = 1 / alpha_0[:, None, None] * (torch.matmul(B, torch.conj(B).permute(0, 2, 1)) - torch.matmul(C,torch.conj(C).permute(0,2,1)))
            Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j
            L, U = torch.linalg.eigh(Gamma)
            Cov_out = U @ torch.diag_embed(1 / L).cfloat() @ U.mH
        return mu_out, Cov_out

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        if self.cov_type == 'DFT':
            mu_out,log_pre = self.decode(z)
            Gamma = torch.diag_embed(torch.exp(log_pre)) + 0j
            return mu_out,Gamma,mu,log_var
        if self.cov_type == 'Toeplitz':
            mu_out, B, C = self.decode(z)
            alpha_0 = B[:, 0, 0]
            Gamma = 1 / alpha_0[:, None,None] * (torch.matmul(B, torch.conj(B).permute(0,2,1)) - torch.matmul(C,torch.conj(C).permute(0,2,1)))
            return mu_out,Gamma, mu, log_var



class my_tra_VAE(nn.Module):
    def __init__(self,cov_type,ld,conv_layer,total_layer,out_channels,k_size,prepro,n_snapshots,device):
        super().__init__()
        rand_matrix = torch.randn(32, 32)
        self.device = device
        self.prepro = prepro
        self.F = torch.zeros((32, 32), dtype=torch.cfloat).to(self.device)
        for m in range(32):
            for n in range(32):
                self.F[m, n] = 1 / torch.sqrt(torch.tensor(32)) * torch.exp(torch.tensor(1j * 2 * math.pi * (m * n) / 32))
        self.B_mask = torch.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.B_mask = self.B_mask[None, :, :].to(self.device)

        self.C_mask = torch.tril(rand_matrix, diagonal=-1)
        self.C_mask[self.C_mask != 0] = 1
        self.C_mask = self.C_mask[None, :, :].to(self.device)
        self.latent_dim = ld
        self.conv_layer = conv_layer
        self.cov_type = cov_type
        self.total_layer = total_layer
        self.out_channels = out_channels
        self.k_size = k_size
        self.n_snapshots = n_snapshots
        if conv_layer > 0:
            step = int(math.floor((out_channels - 2)/conv_layer))
        self.encoder = []
        in_channels = 2
        for i in range(conv_layer-1):
            self.encoder.append(nn.Conv2d(in_channels,in_channels + step,k_size,2,int((k_size-1)/2)))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm2d(in_channels + step))
            in_channels = in_channels + step
        if conv_layer > 0:
            self.encoder.append(nn.Conv2d(in_channels,out_channels,k_size,2,int((k_size-1)/2)))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm2d(out_channels))

        in_linear = 64 * self.n_snapshots
        if conv_layer > 0:
            self.encoder.append(nn.Flatten())
            for i in range(total_layer-conv_layer):
                self.encoder.append(nn.Linear(int(32*16/(4**conv_layer) * out_channels),int(32*16/(4**conv_layer) * out_channels)))
                self.encoder.append(nn.ReLU())
                self.encoder.append(nn.BatchNorm1d(int(32*16/(4**conv_layer) * out_channels)))

            self.fc_mu = nn.Linear(int(32*16 / (4 ** conv_layer) * out_channels), self.latent_dim)
            self.fc_var = nn.Linear(int(32*16 / (4 ** conv_layer) * out_channels), self.latent_dim)
        else:
            self.encoder.append(nn.Linear(int(in_linear),int(out_channels/4 * in_linear)))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm1d(int(out_channels/4 * in_linear)))
            for i in range(1,total_layer - conv_layer - 1):
                self.encoder.append(nn.Linear(int(out_channels / 4 * in_linear),int( out_channels / 4 * in_linear)))
                self.encoder.append(nn.ReLU())
                self.encoder.append(nn.BatchNorm1d(int(out_channels / 4 * in_linear)))

            self.fc_mu = nn.Linear(int(out_channels / 4 * in_linear), self.latent_dim)
            self.fc_var = nn.Linear(int(out_channels / 4 * in_linear), self.latent_dim)
        self.encoder = nn.Sequential(*self.encoder)
        dim_out = 0
        self.decoder_lin = []
        if conv_layer > 0:
            self.decoder_input = nn.Linear(self.latent_dim,int(32 / (2 ** conv_layer) * out_channels))
            for i in range(total_layer-conv_layer):
                self.decoder_lin.append(nn.Linear(int(32/(2**conv_layer) * out_channels),int(32/(2**conv_layer) * out_channels)))
                self.decoder_lin.append(nn.ReLU())
                self.decoder_lin.append(nn.BatchNorm1d(int(32/(2**conv_layer) * out_channels)))
            dim_out = int(32/(2**conv_layer) * out_channels)
        else:
            self.decoder_input = nn.Linear(self.latent_dim, int(out_channels / 4 * in_linear))
            for i in range(1,total_layer - conv_layer - 1):
                self.decoder_lin.append(nn.Linear(int(out_channels / 4 * in_linear), int(out_channels / 4 * in_linear)))
                self.decoder_lin.append(nn.ReLU())
                self.decoder_lin.append(nn.BatchNorm1d(int(out_channels / 4 * in_linear)))
            dim_out = int(out_channels / 4 * in_linear)

        self.decoder_lin = nn.Sequential(*self.decoder_lin)
        self.decoder = []
        if conv_layer > 0:
            dim_out = dim_out / out_channels
        for i in range(conv_layer - 1):
            self.decoder.append(nn.ConvTranspose2d(out_channels, out_channels - step, k_size, 2))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.BatchNorm2d(out_channels - step))
            out_channels = out_channels - step
            dim_out = (dim_out-1) * 2 + (k_size-1) + 1
        #Lout=(Lin−1)×stride−2×padding + dilation×(kernel_size−1) + output_padding + 1


        if conv_layer > 0:
            self.decoder.append(nn.ConvTranspose2d(out_channels, 2, k_size, 2))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.BatchNorm2d(2))
            dim_out = (dim_out - 1) * 2 + (k_size - 1) + 1

        self.decoder = nn.Sequential(*self.decoder)
        if cov_type == 'DFT':
            if self.conv_layer > 0:
                self.final_layer = nn.Linear(int(2 * dim_out), 96)
            else:
                self.final_layer = nn.Linear(int(dim_out), 96)
        if cov_type == 'Toeplitz':
            if self.conv_layer > 0:
                self.final_layer = nn.Linear(int(2 * dim_out),64 + 63)
            else:
                self.final_layer = nn.Linear(int(dim_out), 64 + 63)

    def encode(self, x):
        if (self.cov_type == 'Toeplitz') & (self.prepro == 'DFT'):
            x_new = torch.zeros((x.size())).to(self.device)
            x = x[:, 0, :] + 1j * x[:, 1, :]
            transformed_set = torch.einsum('mn,kn -> km', self.F, x)
            x_new[:, 0, :] = torch.real(transformed_set)
            x_new[:, 1, :] = torch.imag(transformed_set)
            x = x_new
        if self.conv_layer == 0:
            x = nn.Flatten()(x)
        out = self.encoder(x)
        out = nn.Flatten()(out)
        mu, log_var = self.fc_mu(out), self.fc_var(out)
        log_var = (15 + 2.5) / 2 * nn.Tanh()(log_var) + (15 + 2.5) / 2 - 15
        return mu, log_var

    def reparameterize(self, log_var, mu):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mu + eps * std

    def decode(self,z):
        batchsize = z.size(0)
        out = self.decoder_input(z)
        bs = out.size(0)
        out = self.decoder_lin(out)
        if self.conv_layer > 0:
            out = out.view(bs,self.out_channels,-1)
        out = self.decoder(out)
        out = nn.Flatten()(out)

        out = self.final_layer(out)
        if self.cov_type == 'DFT':
            mu_real,mu_imag,log_pre = out.chunk(3,dim=1)
            log_pre = (0.5 + 15) / 2 * nn.Tanh()(log_pre) + (0.5 + 15) / 2 - 0.5
            mu_out = torch.zeros(batchsize,2,32).to(self.device)
            mu_out[:,0,:] = mu_real
            mu_out[:,1,:] = mu_imag
            return mu_out, log_pre

        if self.cov_type == 'Toeplitz':
            mu_real, mu_imag,alpha = out[:,:32],out[:,32:64],out[:,64:]
            alpha_0 = alpha[:, 0][:, None]
            alpha_rest = alpha[:, 1:]
            alpha_0 = torch.exp(alpha_0)
            alpha_intermediate = alpha_0.clone()
            if torch.sum(alpha_intermediate[alpha_0 > 5000]) > 0:
                print('alpha regularized')
            alpha_intermediate[alpha_0 > 5000] = 5000
            alpha_0 = alpha_intermediate.clone()
            alpha_rest = torch.squeeze(alpha_rest)
            alpha_rest = 0.022 * alpha_0 * nn.Tanh()(alpha_rest)
            alpha_rest = torch.complex(alpha_rest[:, :31], alpha_rest[:, 31:])
            Alpha = torch.cat((alpha_0, alpha_rest), dim=1)
            Alpha_prime = torch.cat((torch.zeros(batchsize, 1).to(self.device), Alpha[:, 1:].flip(1)), dim=1)
            values = torch.cat((Alpha, Alpha[:, 1:].flip(1)), dim=1)
            i, j = torch.ones(32, 32).nonzero().T
            values = values[:, j - i].reshape(batchsize, 32, 32)
            B = values * self.B_mask

            values_prime = torch.cat((Alpha_prime, Alpha_prime[:, 1:].flip(1)), dim=1)
            i, j = torch.ones(32, 32).nonzero().T
            values_prime2 = values_prime[:, j - i].reshape(batchsize, 32, 32)
            C = torch.conj(values_prime2 * self.C_mask)
            mu_out = torch.zeros(batchsize,2,32).to(self.device)
            mu_out[:,0,:] = mu_real
            mu_out[:,1,:] = mu_imag
            return mu_out,B,C

    def estimating(self,x,estimated_snapshot):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        if self.cov_type == 'DFT':
            mu_out,log_pre = self.decode(z)
            mu_out = mu_out[:,0,:] + 1j * mu_out[:,1,:]
            Cov_out = torch.diag_embed(1 / (torch.exp(log_pre))) + 0j
        if self.cov_type == 'Toeplitz':
            mu_out,B,C = self.decode(z)
            mu_out = mu_out[:, 0, :] + 1j * mu_out[:, 1, :]
            alpha_0 = B[:, 0, 0]
            Gamma = 1 / alpha_0[:, None, None] * (torch.matmul(B, torch.conj(B).permute(0, 2, 1)) - torch.matmul(C,torch.conj(C).permute(0,2,1)))
            Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j
            L, U = torch.linalg.eigh(Gamma)
            Cov_out = U @ torch.diag_embed(1 / L).cfloat() @ U.mH
        return mu_out, Cov_out

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        if self.cov_type == 'DFT':
            mu_out,log_pre = self.decode(z)
            Gamma = torch.diag_embed(torch.exp(log_pre)) + 0j
            return mu_out,Gamma,mu,log_var
        if self.cov_type == 'Toeplitz':
            mu_out, B, C = self.decode(z)
            alpha_0 = B[:, 0, 0]
            Gamma = 1 / alpha_0[:, None,None] * (torch.matmul(B, torch.conj(B).permute(0,2,1)) - torch.matmul(C,torch.conj(C).permute(0,2,1)))
            return mu_out,Gamma, mu, log_var
