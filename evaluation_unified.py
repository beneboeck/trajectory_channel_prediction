import math
import torch
import training_unified as tr
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from torch.utils.data import DataLoader
from mmd_utils import *

def eval_val(model_type,setup,model,dataloader_val,cov_type, lamba,device, dir_path):
    iterator = iter(dataloader_val)
    samples = iterator.next()
    if cov_type == 'DFT':
        sample = samples[2].to(device)
    else:
        sample = samples[0].to(device)

    if (model_type == 'Trajectory') & (cov_type == 'Toeplitz'):
        out, z, eps, mu_inf, log_var = model(sample)
        mu_out, B_out, C_out = out
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_toeplitz_free_bits(lamba, sample, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)
        m_sigma_squared_prior = torch.mean(1 / torch.exp(logpre_prior)).item()
        m_sigma_squared_inf = torch.mean(torch.exp(log_var)).item()
        std_sigma_squared_prior = torch.std(1 / torch.exp(logpre_prior)).item()
        std_sigma_squared_inf = torch.std(torch.exp(log_var)).item()
        m_alpha_0 = torch.mean(torch.abs(B_out[:,:,0,0])).item()
        std_alpha_0 = torch.std(torch.abs(B_out[:,:,0,0])).item()
        bound = 0.02 * torch.abs(B_out[:,:,0,0])
        n_bound_hits = torch.mean(torch.sum(torch.abs(torch.real(B_out[:,:,1:,0])) > bound[:,:,None],dim=2).float()).item() + torch.mean(torch.sum(torch.abs(torch.imag(B_out[:,:,1:,0])) > bound[:,:,None],dim=2)).item()
        output_stats = [m_sigma_squared_prior,std_sigma_squared_prior,m_sigma_squared_inf,std_sigma_squared_inf,m_alpha_0,std_alpha_0,n_bound_hits]

    if (model_type == 'Trajectory') & (cov_type == 'DFT'):
        out, z, eps, mu_inf, log_var = model(sample)
        mu_out, logpre_out = out
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)
        m_sigma_squared_out = torch.mean(1/torch.exp(logpre_out)).item()
        m_sigma_squared_prior = torch.mean(1/torch.exp(logpre_prior)).item()
        m_sigma_squared_inf = torch.mean(torch.exp(log_var)).item()
        std_sigma_squared_out = torch.std(1/torch.exp(logpre_out)).item()
        std_sigma_squared_prior = torch.std(1/torch.exp(logpre_prior)).item()
        std_sigma_squared_inf = torch.std(torch.exp(log_var)).item()
        output_stats = [m_sigma_squared_prior,std_sigma_squared_prior, m_sigma_squared_inf, std_sigma_squared_inf,m_sigma_squared_out,std_sigma_squared_out]


    if model_type == 'Single':
        sample = sample[:, :, :, -1]
        mu_out, Gamma, mu, log_var = model(sample)
        Risk, RR, KL = tr.risk_free_bits(lamba, sample, mu, log_var, mu_out, Gamma)
        output_stats = []

    if (model_type == 'TraSingle'):
        single_sample = sample[:, :, :, -1]
        mu_out, Gamma, mu, log_var = model(sample)
        Risk, RR, KL = tr.risk_free_bits(lamba, single_sample, mu, log_var, mu_out, Gamma)
        output_stats = []

    if model_type == 'Trajectory':
        NMSE = channel_prediction(setup,model,dataloader_val,15,dir_path,device,'evaluation')
    else:
        NMSE = 0

    return NMSE,Risk,output_stats

def channel_prediction(setup,model,dataloader_val,knowledge,dir_path,device,PHASE):
    cov_type = setup[9]
    NMSE_list = []
    for ind,sample in enumerate(dataloader_val):
        if cov_type == 'DFT':
            sample = sample[2].to(device)
        if cov_type == 'Toeplitz':
            sample = sample[0].to(device)
        predicted_samples, ground_truth = model.predicting(sample, knowledge) # BS,2,N_ANT,SNAPSHOTS - KNOWLEDGE
        NMSE = torch.mean(torch.sum(torch.abs(ground_truth - predicted_samples) ** 2, dim=(1,2,3)) / torch.sum(torch.abs(ground_truth) ** 2,dim=(1,2,3))).detach().to('cpu')
        NMSE_list.append(NMSE)

    #if PHASE == 'testing':
    #    prediction_visualization(setup,ground_truth,predicted_samples,dir_path)
    NMSE = np.mean(np.array(NMSE_list))
    return NMSE


def prediction_visualization(setup,samples,complete_x_list,dir_path):
    cov_type = setup[9]
    if cov_type == 'DFT':
        samples = apply_IDFT(samples)
        complete_x_list = apply_IDFT(complete_x_list)
        samples = torch.tensor(samples)
        complete_x_list = torch.tensor(complete_x_list)

    fig, ax = plt.subplots(4, 6, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(18, 4))

    for n in range(6):
        sample = samples[n, :, :, :]
        sample = sample[None, :, :, :]

        mu_out = complete_x_list[n, :, :]
        mu_out = torch.squeeze(mu_out).to('cpu').detach()
        mu_out = torch.complex(mu_out[0, :], mu_out[1, :])
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


def channel_estimation(model,dataloader_val,sig_n,cov_type,dir_path,device):
    NMSE_list = []
    for ind, samples in enumerate(dataloader_val):
        estimated_snapshot = -1
        if cov_type == 'DFT':
            sample = samples[2].to(device) # BS, 2, N_ANT, SNAPSHOTS
            received_signal = samples[3].to(device)
        if cov_type == 'Toeplitz':
            sample = samples[0].to(device)
            received_signal = samples[1].to(device)
        sample_oi = sample[:,0,:,estimated_snapshot] + 1j * sample[:,1,:,estimated_snapshot] # BS, N_ANT
        received_signal_oi = received_signal[:,0,:,estimated_snapshot] + 1j * received_signal[:,1,:,estimated_snapshot]
        mu_out,Cov_out = model.estimating(sample,estimated_snapshot) # BS,N_ANT complex; BS, N_ANT, N_ANT complex
        print('Frobenius Cov out')
        print(torch.mean(torch.sum(torch.abs(Cov_out)**2,dim=(1,2))))
        L,U = torch.linalg.eigh(Cov_out)
        inv_matrix = U @ torch.diag_embed(1/(L + sig_n ** 2)).cfloat() @ U.mH
        h_hat = mu_out + torch.einsum('ijk,ik->ij', Cov_out @ inv_matrix, (received_signal_oi - mu_out))
        NMSE = torch.mean(torch.sum(torch.abs(sample_oi - h_hat) ** 2, dim=1) / torch.sum(torch.abs(sample_oi) ** 2,dim=1)).detach().to('cpu')
        NMSE_list.append(NMSE)

    NMSE = np.mean(np.array(NMSE_list))
    return NMSE

def channel_estimation_all(model,dataloader_val,sig_n,cov_type,dir_path,device):
    NMSE_list = []
    estimated_snapshot = -1
    for ind, samples in enumerate(dataloader_val):
        if cov_type == 'DFT':
            sample = samples[2].to(device) # BS, 2, N_ANT, SNAPSHOTS
            received_signal = samples[3].to(device)
        if cov_type == 'Toeplitz':
            sample = samples[0].to(device)
            received_signal = samples[1].to(device)
        sample_oi = sample[:,0,:,estimated_snapshot] + 1j * sample[:,1,:,estimated_snapshot] # BS, N_ANT
        NMSE_final = 10e5
        for i in range(16):
            received_signal_oi = torch.mean(received_signal[:,0,:,i:] + 1j * received_signal[:,1,:,i:],dim=2)
            mu_out,Cov_out = model.estimating(sample,estimated_snapshot) # BS,N_ANT complex, BS, N_ANT, N_ANT complex
            L,U = torch.linalg.eigh(Cov_out)
            inv_matrix = U @ torch.diag_embed(1/(L + sig_n ** 2)).cfloat() @ U.mH
            h_hat = mu_out + torch.einsum('ijk,ik->ij', Cov_out @ inv_matrix, (received_signal_oi - mu_out))
            NMSE = torch.mean(torch.sum(torch.abs(sample_oi - h_hat) ** 2, dim=1) / torch.sum(torch.abs(sample_oi) ** 2,dim=1)).detach().to('cpu')
            if NMSE < NMSE_final:
                NMSE_final = NMSE
    NMSE = np.array(NMSE_final)
    return NMSE


def computing_MMD(setup,model,n_iterations,n_permutations,normed,bs_mmd,dataset_val,snapshots,dir_path,device):
    LD = setup[0]
    cov_type = setup[9]
    alpha = 0.05
    batchsize=bs_mmd
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
        #print('test')
        #print(torch.mean(z_samples))
        #print(torch.std(z_samples))
        if (cov_type == 'diagonal') | (cov_type == 'DFT'):
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

        #print(torch.mean(mu_out))
        #print(torch.std(mu_out))
        # here I draw samples from q(z|x) generated by samples2

        latent_rep = model.encode(samples2)
        z, eps, mu_inf, logvar_inf = latent_rep

        z_samples = torch.randn(batchsize, LD, snapshots).to(device)
        z_samples = torch.exp(0.5 * logvar_inf) * z_samples + mu_inf
        z_samples = z_samples.view(batchsize, LD, snapshots)
        output_rep = model.decode(z_samples)

        if (cov_type == 'diagonal') | (cov_type == 'DFT'):
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

        if cov_type == 'DFT':
            mu_out = torch.tensor(apply_IDFT(mu_out)).to(device)
            mu_out2 = torch.tensor(apply_IDFT(mu_out2)).to(device)
            samples = torch.tensor(apply_IDFT(samples)).to(device)
            samples2 = torch.tensor(apply_IDFT(samples2)).to(device)

        mu_out_MMD = mu_out.reshape(batchsize, -1)
        samples_MMD = samples.view(batchsize, -1)
        mu_out2_MMD = mu_out2.reshape(batchsize, -1)
        samples2_MMD = samples.view(batchsize, -1)

        del mu_out2, mu_out, samples, samples2
        torch.cuda.empty_cache()

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

def computing_LS_sample_covariance_estimator(dataset_val,sig_n):
    h_hat_LS = dataset_val.y[:,0,:,-1] + 1j * dataset_val.y[:,1,:,-1]
    h = dataset_val.h[:,0,:,-1] + 1j * dataset_val.h[:,1,:,-1]
    NMSE_LS = torch.mean(torch.linalg.norm(h - h_hat_LS,dim = 1) ** 2)/torch.mean(torch.linalg.norm(h,dim=1)**2)

    sMean = torch.mean(h_hat_LS,dim=0)
    sCov_y = torch.mean(torch.einsum('ij,ik->ijk',(h_hat_LS - sMean),torch.conj(h_hat_LS - sMean)),dim=0)
    inv_matrix = torch.linalg.inv(sCov_y)
    h_hat_sCov = sMean[None,:] + torch.einsum('ij,kj->ki',(sCov_y - sig_n ** 2 * torch.eye(32,32,dtype=torch.cfloat)) @ inv_matrix,(h_hat_LS - sMean[None,:]))
    NMSE_sCov = torch.mean(torch.linalg.norm(h - h_hat_sCov, dim=1) ** 2) / torch.mean(torch.linalg.norm(h, dim=1) ** 2)

    return NMSE_LS,NMSE_sCov

def computing_LS_sample_covariance_estimator_all(dataset_val,sig_n):
    NMSE_LS_end = 10e5
    NMSE_sCov_end = 10e5
    for i in range(16):
        h_hat_LS = torch.mean(dataset_val.y[:,0,:,i:] + 1j * dataset_val.y[:,1,:,i:],dim=2)
        h = dataset_val.h[:,0,:,-1] + 1j * dataset_val.h[:,1,:,-1]
        NMSE_LS = torch.mean(torch.linalg.norm(h - h_hat_LS,dim = 1) ** 2)/torch.mean(torch.linalg.norm(h,dim=1)**2)

        h_hat_LS_single = dataset_val.y[:,0,:,-1] + 1j * dataset_val.y[:,1,:,-1]
        sMean = torch.mean(h_hat_LS_single,dim=0)
        sCov_y = torch.mean(torch.einsum('ij,ik->ijk',(h_hat_LS_single - sMean),torch.conj(h_hat_LS_single - sMean)),dim=0)
        inv_matrix = torch.linalg.inv(sCov_y)
        h_hat_sCov = sMean[None,:] + torch.einsum('ij,kj->ki',(sCov_y - sig_n ** 2 * torch.eye(32,32,dtype=torch.cfloat)) @ inv_matrix,(h_hat_LS - sMean[None,:]))
        NMSE_sCov = torch.mean(torch.linalg.norm(h - h_hat_sCov, dim=1) ** 2) / torch.mean(torch.linalg.norm(h, dim=1) ** 2)
        if NMSE_LS < NMSE_LS_end:
            NMSE_LS_end = NMSE_LS
        if NMSE_sCov < NMSE_sCov_end:
            NMSE_sCov_end = NMSE_sCov
    return NMSE_LS_end,NMSE_sCov_end

def keep_last(dataloader,knowledge,device):
    NMSE_list = []
    for ind,sample in enumerate(dataloader):
        sample = sample[0].to(device)
        n_snaps = sample.size(3)
        ground_truth = sample[:,:,:,knowledge:]
        predicted_samples = sample[:,:,:,knowledge-1:knowledge].repeat(1,1,1,n_snaps - knowledge)
        NMSE = torch.mean(torch.sum(torch.abs(ground_truth - predicted_samples) ** 2, dim=(1,2,3)) / torch.sum(torch.abs(ground_truth) ** 2,dim=(1,2,3))).detach().to('cpu')
        NMSE_list.append(NMSE)
    NMSE = np.mean(np.array(NMSE_list))
    return NMSE