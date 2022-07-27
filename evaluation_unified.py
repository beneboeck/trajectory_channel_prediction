import math
import torch
import training_unified as tr
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def eval_val(setup,model,dataloader_val,cov_type, lamba,device, dir_path):

    iterator = iter(dataloader_val)
    samples = iterator.next()
    sample = samples[0].to(device)

    if (cov_type == 'Toeplitz'):
        out, z, eps, mu_inf, log_var = model(sample)
        mu_out, B_out, C_out = out
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_toeplitz_free_bits(lamba, sample, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)

    if (cov_type == 'diagonal') | (cov_type == 'DFT'):
        out, z, eps, mu_inf, log_var = model(sample)
        mu_out, logpre_out = out
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)


    NMSE = channel_prediction(setup,model,dataloader_val,16,dir_path,device,'evaluation')
    return NMSE, Risk


def channel_prediction(setup,model,dataloader_val,knowledge,dir_path,device,PHASE):
    LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type = setup
    NMSE_list = []
    for ind,sample in enumerate(dataloader_val):
        samples = sample[0].to(device)
        n_units = int(samples.size(3))

        # encoding
        batchsize = samples.size(0)
        z = torch.zeros(batchsize, LD, n_units).to(device)
        hidden_state = torch.zeros(batchsize, LD, n_units).to(device)
        z_init = torch.ones(batchsize, LD).to(device)  # zeros instead of ones in the spirit of Glow
        if memory > 0:
            x_start = torch.ones(batchsize, 2, 32, memory).to(device)
        if memory > 0:
            x_input = torch.cat((x_start, samples[:, :, :, 0][:, :, :, None]), dim=3)
        else:
            x_input = samples[:, :, :, :1]
        mu_z, logvar_z, hidden_state[:, :, 0] = model.encoder[0](x_input, z_init, z_init)
        z[:, :, 0] = mu_z
        for i in range(1, memory):
            x_input = torch.cat((x_start[:, :, :, :memory - i], samples[:, :, :, :i + 1]), dim=3)
            z_input = z[:, :, i - 1].clone()
            mu_z, logvar_z, hidden_state[:, :, i] = model.encoder[i](x_input, z_input, hidden_state[:, :, i - 1].clone())
            # logpre_out_local[logpre_out_local > 9] = 9
            z[:, :, i] = mu_z
        for unit in range(memory, knowledge):
            z_input = z[:, :, unit - 1].clone()
            x_input = samples[:, :, :, unit - memory:unit + 1]
            mu_z, logvar_z, hidden_state[:, :, unit] = model.encoder[unit](x_input, z_input,hidden_state[:, :, unit - 1].clone())
            z[:, :, unit] = mu_z

        # prior
        z_input = z[:,:,knowledge-1].clone()
        for idx in range(knowledge,n_units):
            z_local,_,hidden_state[:,:,idx] = model.prior_model[idx](z_input,hidden_state[:,:,idx-1].clone())
            z[:,:,knowledge] = z_local
            z_input = z_local.clone()

        # prediction
        x_list = torch.zeros(samples.size(0),2,32,n_units-knowledge).to(device)
        for idx in range(knowledge,n_units):
            x_local = model.decoder[idx](z[:,:,idx-memory:idx+1])[0]
            x_list[:,:,:,(idx-knowledge):(idx-knowledge+1)] = x_local
            predicted_samples = samples[:,:,:,knowledge:]
            complete_x_list = torch.cat((samples[:, :, :, :knowledge], x_list), dim=3)
            NMSE_list.append(torch.mean(torch.sum((predicted_samples - x_list) ** 2, dim=(1, 2, 3)) / torch.sum(predicted_samples ** 2,dim=(1, 2, 3))).detach().to('cpu'))

    if PHASE == 'testing':
        prediction_visualization(setup,samples,complete_x_list,dir_path)
    NMSE = np.mean(np.array(NMSE_list))
    return NMSE


def prediction_visualization(setup,samples,complete_x_list,dir_path):
    cov_type = setup[-1]
    if cov_type == 'DFT':
        samples = apply_IDFT(samples)
        complete_x_list = apply_IDFT(complete_x_list)
        samples = torch.tensor(samples)
        complete_x_list = torch.tensor(complete_x_list)

    fig, ax = plt.subplots(4, 6, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(18, 4))

    for n in range(6):
        sample = samples[n, :, :, :]
        sample = sample[None, :, :, :]

        mu_out = complete_x_list[n, :, :, :]
        mu_out = torch.squeeze(mu_out).to('cpu').detach()
        mu_out = torch.complex(mu_out[0, :, :], mu_out[1, :, :])
        abs_out = torch.abs(mu_out)
        angle_out = torch.angle(mu_out)

        sample = sample.to('cpu')
        sample = torch.squeeze(sample)
        sample = torch.complex(sample[0, :, :], sample[1, :, :])
        abs_sample = torch.abs(sample)
        angle_sample = torch.angle(sample)

        ax[int(np.floor(n / 3) * 2), int(n % 3 * 2)].imshow(abs_sample.numpy(), cmap='hot')
        ax[int(np.floor(n / 3) * 2), int(n % 3 * 2)].set_xticks([])
        ax[int(np.floor(n / 3) * 2), int(n % 3 * 2)].set_yticks([])

        ax[int(np.floor(n / 3) * 2), int(n % 3 * 2) + 1].imshow(abs_out.numpy(), cmap='hot')
        ax[int(np.floor(n / 3) * 2), int(n % 3 * 2) + 1].set_xticks([])
        ax[int(np.floor(n / 3) * 2), int(n % 3 * 2) + 1].set_yticks([])

        ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2)].imshow(angle_sample.numpy(), cmap='hot')
        ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2)].set_xticks([])
        ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2)].set_yticks([])

        ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2 + 1)].imshow(angle_out.numpy(), cmap='hot')
        ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2 + 1)].set_xticks([])
        ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2 + 1)].set_yticks([])

    fig.suptitle('Real Domain (antennas)- NW:original abs,NE:estimated abs,SW:original phase,SE:estimated phase')

    fig.savefig(dir_path + '/heat_map_for_prediction.png', dpi=300)
    plt.close('all')

def channel_estimation(setup,model,dataloader_val,sig_n,dir_path,device):
    LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type = setup
    NMSE_list = []

    #encoding
    for ind, sample in enumerate(dataloader_val):
        sample,noisy_sample = sample
        sample = sample.to(device)
        noisy_sample = noisy_sample.to(device)
        snapshots = int(sample.size(3))
        batchsize = sample.size(0)
        z = torch.zeros(batchsize, LD, snapshots).to(device)
        hidden_state = torch.zeros(batchsize, LD, snapshots).to(device)
        z_init = torch.ones(batchsize,LD).to(device)  # zeros instead of ones in the spirit of Glow
        if memory > 0:
            x_start = torch.ones(batchsize, 2, 32, memory).to(device)

        if memory > 0:
            x_input = torch.cat((x_start, sample[:, :, :, 0][:, :, :, None]), dim=3)
        else:
            x_input = sample[:, :, :, :1]
        mu_z, logvar_z, hidden_state[:, :, 0] = model.encoder[0](x_input, z_init, z_init)
        z[:, :, 0] = mu_z

        for i in range(1, memory):
            x_input = torch.cat((x_start[:, :, :, :memory - i], sample[:, :, :, :i + 1]), dim=3)
            z_input = z[:, :, i - 1].clone()
            mu_z, logvar_z, hidden_state[:, :, i] = model.encoder[i](x_input, z_input, hidden_state[:, :, i - 1].clone())
            z[:, :, i] = mu_z

        for unit in range(memory, snapshots):
            z_input = z[:, :, unit - 1].clone()
            x_input = sample[:, :, :, unit - memory:unit + 1]
            mu_z, logvar_z, hidden_state[:, :, unit] = model.encoder[unit](x_input, z_input,hidden_state[:, :, unit - 1].clone())
            z[:, :, unit] = mu_z

        #decoding

        batchsize = z.size(0)
        mu_out = torch.zeros(batchsize, 2, 32, snapshots).to(device)
        z_init = torch.ones(batchsize, LD, memory).to(device)
        if cov_type == 'Toeplitz':
            B_out = torch.zeros(batchsize, snapshots, 32, 32, dtype=torch.cfloat).to(device)
            C_out = torch.zeros(batchsize, snapshots, 32, 32, dtype=torch.cfloat).to(device)
        else:
            logpre_out = torch.zeros(batchsize, 32, snapshots).to(device)

        for i in range(memory):
            z_input = torch.cat((z_init[:, :, :memory - i], z[:, :, :i + 1]), dim=2)
            if cov_type == 'Toeplitz':
                mu_out_local, B_out_local, C_out_local = model.decoder[i](z_input)
                mu_out[:, :, :, i:(i + 1)], B_out[:, i:(i + 1), :, :], C_out[:, i:(i + 1), :,:] = mu_out_local, B_out_local, C_out_local
            else:
                mu_out_local, logpre_local = model.decoder[i](z_input)
                mu_out[:, :, :, i:(i + 1)], logpre_out[:, :, i:i + 1] = mu_out_local, logpre_local
            # logpre_out_local[logpre_out_local > 9] = 9

        for unit in range(memory, snapshots):
            z_input = z[:, :, unit - memory:unit + 1].clone()
            if cov_type == 'Toeplitz':
                mu_out_local, B_out_local, C_out_local = model.decoder[unit](z_input)
                mu_out[:, :, :, unit:unit + 1] = mu_out_local
                B_out[:, unit:unit + 1, :, :] = B_out_local
                C_out[:, unit:unit + 1, :, :] = C_out_local
            else:
                mu_out_local, logpre_local = model.decoder[unit](z_input)
                mu_out[:, :, :, unit:unit + 1] = mu_out_local
                logpre_out[:, :, unit:unit + 1] = logpre_local
                # logpre_out_local[logpre_out_local > 9] = 9

        x_compl = torch.complex(sample[:, 0, :, :], sample[:, 1, :, :]).permute(0, 2, 1)
        mu_compl = torch.complex(mu_out[:, 0, :, :], mu_out[:, 1, :, :]).permute(0, 2, 1)
        noisy_sample_compl = torch.complex(noisy_sample[:, 0, :, :], noisy_sample[:, 1, :, :]).permute(0, 2,1)  # BS, SNAPSHOTS, ANTENNAS
        if cov_type == 'Toeplitz':
            alpha_0 = B_out[:, :, 0, 0]
            if len(alpha_0.size()) == 2:
                Gamma = 1 / alpha_0[:, :, None, None] * (torch.matmul(B_out, torch.conj(B_out).permute(0, 1, 3, 2)) - torch.matmul(C_out,torch.conj(C_out).permute(0,1,3,2)))
            if len(alpha_0.size()) == 1:
                Gamma = 1 / alpha_0[None, :, None, None] * (torch.matmul(B_out, torch.conj(B_out).permute(0, 1, 3, 2)) - torch.matmul(C_out,torch.conj(C_out).permute(0,1,3,2)))

            Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j

            L,U = torch.linalg.eigh(Gamma)
            Cov_out = U @ torch.diag_embed(1/L).cfloat() @ U.mH
            L_noisy = (L + (sig_n**2 * torch.eye(32,32).to(device))[None,None,:,:]).cfloat()

            h_hat = mu_compl + torch.einsum('ijkl,ijl->ijk',Cov_out @ (U @ torch.diag_embed(L_noisy) @ U.mH), (noisy_sample_compl - mu_compl))
            h_hat_last = h_hat[:,-1,:]


        if (cov_type == 'diagonal') | (cov_type == 'DFT'):
            Cov_out = torch.diag_embed(1/(torch.exp(logpre_out.permute(0,2,1)))).cfloat()
            inv_matrix = 1/Cov_out + (sig_n**2 * torch.eye(32,32).to(device)).cfloat()[None,None,:,:]

            print('hier')
            print(mu_compl.size())
            print(Cov_out.size())
            print(inv_matrix.size())
            print(noisy_sample_compl.size())

            h_hat = mu_compl + torch.einsum('ijkl,ijl->ijk',Cov_out @ inv_matrix, (noisy_sample_compl - mu_compl))
            h_hat_last = h_hat[:, -1, :]

        if cov_type == 'DFT':
            h_hat = torch.tensor(apply_IDFT(h_hat.permute(0,2,1))).permute(0,2,1)
            h_hat_last = h_hat[:,-1,:]

        h_last = x_compl[:, -1, :]

        NMSE_list.append(torch.mean(torch.sum(torch.abs(h_last - h_hat_last) ** 2, dim=(1)) / torch.sum(torch.abs(h_last) ** 2,dim=(1))).detach().to('cpu'))

    NMSE = np.mean(np.array(NMSE_list))
    return NMSE