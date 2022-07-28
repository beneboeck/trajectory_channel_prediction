import math
import torch
import training_unified as tr
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from torch.utils.data import DataLoader
from mmd_utils import *

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

        print('channel estimation testing')
        print(torch.max(torch.abs(z)))

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
            L_noisy = (torch.diag_embed(L) + (sig_n**2 * torch.eye(32,32).to(device))[None,None,:,:]).cfloat()

            h_hat = mu_compl + torch.einsum('ijkl,ijl->ijk',Cov_out @ (U @ L_noisy @ U.mH), (noisy_sample_compl - mu_compl))
            h_hat_last = h_hat[:,-1,:]


        if (cov_type == 'diagonal') | (cov_type == 'DFT'):
            Cov_out = torch.diag_embed(1/(torch.exp(logpre_out.permute(0,2,1)))).cfloat()
            inv_matrix = 1/Cov_out + (sig_n**2 * torch.eye(32,32).to(device)).cfloat()[None,None,:,:]
            print('estimation test')
            print(torch.max(torch.abs(Cov_out)))
            print(torch.max(torch.abs(inv_matrix)))
            print(torch.min(torch.abs(inv_matrix)))

            h_hat = mu_compl + torch.einsum('ijkl,ijl->ijk',Cov_out @ inv_matrix, (noisy_sample_compl - mu_compl))
            print('hier')
            print(h_hat.size())
            h_hat_last = h_hat[:, -1, :]

        if cov_type == 'DFT':
            h_hat_realed = torch.zeros((h_hat.size(0),2,h_hat.size(1),h_hat.size(2)),dtype=torch.cfloat)
            h_hat_realed[:,0,:,:] = torch.real(h_hat)
            h_hat_realed[:,1,:,:] = torch.imag(h_hat)
            h_hat = torch.tensor(apply_IDFT(h_hat_realed.permute(0,1,3,2))).permute(0,1,3,2)
            h_hat = h_hat[:,0,:,:] + 1j * h_hat[:,1,:,:]
            h_hat_last = h_hat[:,-1,:]

        h_last = x_compl[:, -1, :]

        print(torch.max(torch.abs(mu_compl)))

        NMSE_list.append(torch.mean(torch.sum(torch.abs(h_last - h_hat_last) ** 2, dim=1) / torch.sum(torch.abs(h_last) ** 2,dim=1)).detach().to('cpu'))

    NMSE = np.mean(np.array(NMSE_list))
    return NMSE


def computing_MMD(setup,model,n_iterations,n_permutations,normed,dataset_val,snapshots,dir_path,device):
    LD, memory, rnn_bool, en_layer, en_width, pr_layer, pr_width, de_layer, de_width, cov_type = setup
    alpha = 0.05
    batchsize=1000
    H = np.zeros(n_iterations)
    H2 = np.zeros(n_iterations)

    for g in range(n_iterations):
        if g%100 == 0:
            print(f'iteration: {g}')
        #print('new iteration')

        loader = DataLoader(dataset_val, batch_size=batchsize, shuffle=True)
        iterator = iter(loader)

            # samples are for the comparison with Quadriga Data

        samples = iterator.next()
        samples = samples[0].to(device)

            # samples2 are for the generating latent distributions q(z|x) for MMD inf

        samples2 = iterator.next()
        samples2 = samples2[0].to(device)

        # here I create completely new data

        z_samples = model.sample_from_prior(batchsize)
        print('test')
        print(torch.mean(z_samples))
        print(torch.std(z_samples))
        if cov_type == 'diagonal':
            mu_out, logpre_out = model.decode(z_samples)  # (BS,2,32,S) , (BS,1,32,S)
            new_gauss = 1 / (torch.sqrt(torch.tensor(2).to(device))) * (torch.randn(mu_out.size()[0], mu_out.size()[2], mu_out.size()[3]).to(device) + 1j * torch.randn(mu_out.size()[0], mu_out.size()[2],mu_out.size()[3]).to(device))  # (BS,32,S) complex
            new_var = new_gauss * torch.exp(-0.5 * torch.squeeze(logpre_out))  # (BS,32,S)
            new_var_new = torch.zeros(mu_out.size()).to(device)  # (BS,2,32,S)
            new_var_new[:, 0, :, :] = torch.real(new_var)
            new_var_new[:, 1, :, :] = torch.imag(new_var)
            mu_out = new_var_new + mu_out

        if cov_type == 'Toeplitz':

            mu_out, B_out, C_out = model.decode(z_samples)  # (BS,2,32,S) , (BS,S,32,32) complex, (BS,S,32,32) compplex
            alpha_0 = B_out[:, :, 0, 0]
            Gamma = 1 / alpha_0[:, :, None, None] * (torch.matmul(B_out, torch.conj(B_out).permute(0, 1, 3, 2)) - torch.matmul(C_out,torch.conj(C_out).permute(0,1,3,2)))
            Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j
            L_G, U = torch.linalg.eigh(Gamma)  # (BS,S,32) complex,(BS,S,32,32) complex
            sqrt_L_C = torch.sqrt(1 / (L_G))
            new_gauss2_compl = 1 / (torch.sqrt(torch.tensor(2).to(device))) * (torch.randn(mu_out.size()[0], mu_out.size()[2], mu_out.size()[3]).to(device) + 1j * torch.randn(mu_out.size()[0], mu_out.size()[2],mu_out.size()[3]).to(device))  # (BS,32,S) compplex
            mu_out_new = sqrt_L_C.permute(0, 2, 1) * new_gauss2_compl  # (BS,32,S) compl
            mu_out_new = torch.einsum('ijkl,ilj->ikj', U, mu_out_new)  # (BS,32,S) complex
            new_var = torch.zeros(mu_out.size()).to(device)
            new_var[:, 0, :, :] = torch.real(mu_out_new)
            new_var[:, 1, :, :] = torch.imag(mu_out_new)
            mu_out = new_var + mu_out

            del B_out, C_out, alpha_0, Gamma, L_G, U, sqrt_L_C, new_gauss2_compl, mu_out_new, new_var
            torch.cuda.empty_cache()

        print(torch.mean(mu_out))
        print(torch.std(mu_out))
        # here I draw samples from q(z|x) generated by samples2

        latent_rep = model.encode(samples2)
        z, eps, mu_inf, logvar_inf = latent_rep

        z_samples = torch.randn(batchsize, LD, snapshots).to(device)
        z_samples = torch.exp(0.5 * logvar_inf) * z_samples + mu_inf
        z_samples = z_samples.view(batchsize, LD, snapshots)
        output_rep = model.decode(z_samples)

        if cov_type == 'diagonal':
            mu_out2, logpre_out2 = output_rep  # (BS,2,32,S) , (BS,1,32,S)
            new_gauss = 1 / (torch.sqrt(torch.tensor(2).to(device))) * (torch.randn(mu_out2.size()[0], mu_out2.size()[2], mu_out2.size()[3]).to(device) + 1j * torch.randn(mu_out2.size()[0], mu_out2.size()[2],mu_out2.size()[3]).to(device))  # (BS,32,S) complex
            new_var = new_gauss * torch.exp(-0.5 * torch.squeeze(logpre_out2))  # (BS,32,S)
            new_var_new = torch.zeros(mu_out2.size()).to(device)  # (BS,2,32,S)
            new_var_new[:, 0, :, :] = torch.real(new_var)
            new_var_new[:, 1, :, :] = torch.imag(new_var)
            mu_out2 = new_var_new + mu_out2

        if cov_type == 'Toeplitz':
            mu_out2, B_out2, C_out2 = output_rep  # (BS,2,32,S) , (BS,S,32,32) complex, (BS,S,32,32) compplex
            alpha_02 = B_out2[:, :, 0, 0]
            Gamma2 = 1 / alpha_02[:, :, None, None] * (torch.matmul(B_out2, torch.conj(B_out2).permute(0, 1, 3, 2)) - torch.matmul(C_out2,torch.conj(C_out2).permute(0,1,3,2)))
            Gamma2[torch.abs(torch.imag(Gamma2)) < 10 ** (-5)] = torch.real(Gamma2[torch.abs(torch.imag(Gamma2)) < 10 ** (-5)]) + 0j
            L_G2, U2 = torch.linalg.eigh(Gamma2)  # (BS,S,32) complex,(BS,S,32,32) complex
            sqrt_L_C2 = torch.sqrt(1 / (L_G2))
            new_gauss2_compl2 = 1 / (torch.sqrt(torch.tensor(2).to(device))) * (torch.randn(mu_out2.size()[0], mu_out2.size()[2], mu_out2.size()[3]).to(device) + 1j * torch.randn(mu_out2.size()[0], mu_out2.size()[2],mu_out2.size()[3]).to(device))  # (BS,32,S) compplex
            mu_out_new2 = sqrt_L_C2.permute(0, 2, 1) * new_gauss2_compl2  # (BS,32,S) compl
            mu_out_new2 = torch.einsum('ijkl,ilj->ikj', U2, mu_out_new2)  # (BS,32,S) complex
            new_var2 = torch.zeros(mu_out2.size()).to(device)
            new_var2[:, 0, :, :] = torch.real(mu_out_new2)
            new_var2[:, 1, :, :] = torch.imag(mu_out_new2)
            mu_out2 = new_var2 + mu_out2
            del B_out2, C_out2, alpha_02, Gamma2, L_G2, U2, sqrt_L_C2, new_gauss2_compl2, mu_out_new2, new_var2
            torch.cuda.empty_cache()



        mu_out2 = mu_out2.detach()
        mu_out = mu_out.detach()
        samples = samples.detach()
        samples2 = samples2.detach()

        mu_out_MMD = mu_out.reshape(batchsize, -1)
        samples_MMD = samples.view(batchsize, -1)
        mu_out2_MMD = mu_out2.reshape(batchsize, -1)
        samples2_MMD = samples.view(batchsize, -1)

        del mu_out2, mu_out, samples, samples2
        torch.cuda.empty_cache()

            # valentina_mu_out = np.array(mu_out_MMD.to('cpu'))
            # valentina_samples = np.array(samples_MMD.to('cpu'))
            # valentina_mu_out2 = np.array(mu_out2_MMD.to('cpu'))
            # valentina_samples2 = np.array(samples2_MMD.to('cpu'))

            # np.savetxt('/home/ga42kab/lrz-nashome/mu_out.txt',valentina_mu_out)
            # np.savetxt('/home/ga42kab/lrz-nashome/samples.txt', valentina_samples)
            # np.savetxt('/home/ga42kab/lrz-nashome/mu_out2.txt', valentina_mu_out2)
            # np.savetxt('/home/ga42kab/lrz-nashome/samples2.txt', valentina_samples2)

        if normed == True:
            mu_out_MMD = mu_out_MMD / torch.linalg.norm(mu_out_MMD, axis=1, keepdims=True)
            samples_MMD = samples_MMD / torch.linalg.norm(samples_MMD, dim=1, keepdim=True)
            mu_out2_MMD = mu_out2_MMD / torch.linalg.norm(mu_out2_MMD, dim=1, keepdim=True)
            samples2_MMD = samples2_MMD / torch.linalg.norm(samples2_MMD, dim=1, keepdim=True)

        Dxy = Pdist2(samples_MMD, mu_out_MMD)
        sigma0 = Dxy.median()

        del Dxy
        torch.cuda.empty_cache()

        Dxy2 = Pdist2(samples2_MMD, mu_out2_MMD)
        sigma02 = Dxy2.median()

            # print('###########')
            # code.interact(local=locals())

        del Dxy2
        torch.cuda.empty_cache()

        mmd_value, Kxyxy, mmd_var = MMDu(samples_MMD, mu_out_MMD, sigma0=sigma0)
        mmd_value2, Kxyxy2, mmd_var2 = MMDu(samples2_MMD, mu_out2_MMD, sigma0=sigma02)

        mmd_vector = np.zeros(n_permutations)
        count = 0
        mmd_vector2 = np.zeros(n_permutations)
        count2 = 0
        nxy = 2 * batchsize

        for i in range(n_permutations):
            ind = np.random.choice(nxy, nxy, replace=False)
                # divide into new X, Y
            indx = ind[:batchsize]
            indy = ind[batchsize:]

                # take the part of the matrix that corresponds to the decision
            Kx = Kxyxy[np.ix_(indx, indx)]
            Ky = Kxyxy[np.ix_(indy, indy)]
            Kxy = Kxyxy[np.ix_(indx, indy)]

            Kx2 = Kxyxy2[np.ix_(indx, indx)]
            Ky2 = Kxyxy2[np.ix_(indy, indy)]
            Kxy2 = Kxyxy2[np.ix_(indx, indy)]

            TEMP = eval_mmd2(Kx, Ky, Kxy)
            mmd_vector[i] = TEMP[0]
            TEMP2 = eval_mmd2(Kx2, Ky2, Kxy2)
            mmd_vector2[i] = TEMP2[0]

            if mmd_vector[i] > mmd_value:
                count = count + 1
            if mmd_vector2[i] > mmd_value2:
                count2 = count2 + 1

            if count > np.ceil(n_permutations * alpha):
                h = 0
            else:
                h = 1

            if count2 > np.ceil(n_permutations * alpha):
                h2 = 0
            else:
                h2 = 1

            if (count > np.ceil(n_permutations * alpha)) & (count2 > np.ceil(n_permutations * alpha)):
                break

        H[g] = h
        H2[g] = h2

    TPR1 = H.sum() / n_iterations
    TPR2 = H2.sum() / n_iterations
    return TPR1, TPR2