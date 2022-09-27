import math
import torch
import evaluation_unified as ev

def risk_toeplitz_free_bits(lamba,x,z,log_var,mu_out,B,C,mu_prior,logpre_prior,eps):
    x_compl = torch.complex(x[:,0,:,:],x[:,1,:,:]).permute(0,2,1)
    mu_compl = torch.complex(mu_out[:,0,:,:],mu_out[:,1,:,:]).permute(0,2,1)
    alpha_0 = B[:, :, 0, 0]
    if len(alpha_0.size()) == 2:
        Gamma = 1 / alpha_0[:, :, None, None] * (torch.matmul(B, torch.conj(B).permute(0, 1, 3, 2)) - torch.matmul(C,torch.conj(C).permute(0,1,3,2)))
    if len(alpha_0.size()) == 1:
        Gamma = 1 / alpha_0[None, :, None, None] * (torch.matmul(B, torch.conj(B).permute(0, 1, 3, 2)) - torch.matmul(C,torch.conj(C).permute(0,1,3,2)))

    Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j
    M, pivots = torch.lu(Gamma)
    P, L, U = torch.lu_unpack(M, pivots)
    diagU = torch.diagonal(U, dim1=2, dim2=3)
    log_detGamma = torch.sum(torch.log(torch.abs(diagU)), dim=2)
    argument = torch.einsum('ijl,ijl->ij', torch.conj(x_compl - mu_compl), torch.einsum('ijkl,ijl->ijk', Gamma, x_compl - mu_compl))
    RR = torch.real(torch.mean(- torch.sum(log_detGamma, dim=1) + torch.sum(argument, dim=1)))
    #IR_term = - 0.5 * eps**2 - 0.5 * log_var
    IR_term = -0.5 * (log_var + 1)
    PR_term = 0.5 * (- logpre_prior + logpre_prior.exp() * (z - mu_prior)**2)

    KL = torch.mean(torch.sum(torch.max(lamba,IR_term + PR_term),dim=(1,2)))

    return RR + KL,RR,KL

def risk_diagonal_free_bits(lamba,x,z,log_var,mu_out,log_pre_out,mu_prior,logpre_prior,eps):
    x_compl = torch.complex(x[:,0,:,:],x[:,1,:,:])
    mu_compl = torch.complex(mu_out[:,0,:,:],mu_out[:,1,:,:])
    RR = torch.mean( torch.sum( - log_pre_out + log_pre_out.exp() * torch.abs(x_compl - mu_compl)**2 ,dim=(1,2)))
    #IR_term = - 0.5 * eps**2 - 0.5 * log_var
    IR_term = -0.5 * (log_var + 1)
    PR_term = 0.5 * (- logpre_prior + logpre_prior.exp() * (z - mu_prior)**2)
    KL = torch.mean(torch.sum(torch.max(lamba,IR_term + PR_term),dim=(1,2)))
    #KL = torch.max(lamba,torch.mean(torch.sum( IR_term + PR_term, dim=(1, 2))))

    return RR + KL,RR,KL

def risk_free_bits(lamba,x,mu,log_var,mu_out,Gamma):
    x = torch.complex(x[:,0,:], x[:,1,:])
    mu_out = torch.complex(mu_out[:,0,:],mu_out[:,1,:])
    Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j
    M, pivots = torch.lu(Gamma)
    P, L, U = torch.lu_unpack(M, pivots)
    diagU = torch.diagonal(U, dim1=1, dim2=2)
    log_detGamma = torch.sum(torch.log(torch.abs(diagU)), dim=1)
    argument = torch.einsum('ij,ij->i', torch.conj(x - mu_out), torch.einsum('ijk,ik->ij', Gamma, x - mu_out))
    Rec_err = torch.real(torch.mean(- log_detGamma + argument))
    #KL =  torch.mean(torch.sum(torch.max(lamba,-0.5 * (1 + log_var - mu ** 2 - (log_var).exp())),dim=1))
    KL = torch.max(lamba,torch.mean(torch.sum( -0.5 * (1 + log_var - mu ** 2 - (log_var).exp()), dim=1)))
    return Rec_err + KL,Rec_err,KL

def training_gen_NN(SNR_format,SNR_range,CSI,model_type,setup,lr, cov_type,model, loader,dataloader_val, epochs, lamba,sig_n,sig_n_train,device, log_file,dir_path,n_iterations, n_permutations, normed,bs_mmd, dataset_val, snapshots):

    risk_list= []
    KL_list = []
    RR_list = []
    eval_risk = []
    eval_NMSE = []
    eval_NMSE_estimation = []
    eval_TPR1 = []
    eval_TPR2 = []
    slope = -1.
    lr_adaption = False

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    print('Start Training ')
    log_file.write('\n\nStart Training\n')

    for i in range(epochs):
        print('epoch')
        print(i)
        for ind, samples in enumerate(loader):
            if CSI == 'PERFECT':
                if cov_type == 'DFT':
                    sample_in = samples[2]
                    sample_ELBO = sample_in
                else:
                    sample_in = samples[0]
                    sample_ELBO = sample_in
                sample_in = sample_in.to(device)
                sample_ELBO = sample_ELBO.to(device)
            if CSI == 'NOISY':
                if SNR_format == 'RANGE':
                    SNRs_train = (SNR_range[1] - SNR_range[0]) * torch.rand(samples[0].size(0)) + SNR_range[0] + (SNR_range[1] - SNR_range[0]) / 2
                    x_train = torch.sum(samples[0][:, :, :, -1] ** 2, dim=(1, 2))
                    SNR_eff = 10 ** (SNRs_train / 10)
                    sig_n_train = torch.sqrt(x_train / (32 * SNR_eff))[:, None, None, None]
                else:
                    sig_n_train = samples[4]
                if cov_type == 'DFT':
                    sample_in = samples[2] + sig_n_train / torch.sqrt(torch.tensor(2)) * torch.randn(samples[2].size())
                    sample_ELBO = samples[2]
                else:
                    sample_in = samples[0] + sig_n_train/torch.sqrt(torch.tensor(2)) * torch.randn(samples[0].size())
                    sample_ELBO = samples[0]
                sample_in = sample_in.to(device)
                sample_ELBO = sample_ELBO.to(device)

            if (model_type == 'Trajectory') & (cov_type == 'Toeplitz'):
                out, z, eps, mu_inf, log_var = model(sample_in)
                mu_out, B_out, C_out = out
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_toeplitz_free_bits(lamba, sample_ELBO, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)

            if (model_type == 'Trajectory') & (cov_type == 'DFT'):
                out, z, eps, mu_inf, log_var = model(sample_in)
                mu_out, logpre_out = out
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_diagonal_free_bits(lamba, sample_ELBO, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)

            if (model_type == 'Single'):
                sample_in = sample_in[:,:,:,-1]
                sample_ELBO = sample_ELBO[:,:,:,-1]
                mu_out, Gamma, mu, log_var = model(sample_in)
                Risk, RR, KL = risk_free_bits(lamba,sample_ELBO,mu,log_var,mu_out,Gamma)

            if (model_type == 'TraSingle'):
                single_sample = sample_ELBO[:,:,:,-1]
                mu_out,Gamma,mu, log_var = model(sample_in)
                Risk,RR,KL = risk_free_bits(lamba,single_sample,mu,log_var,mu_out,Gamma)

            optimizer.zero_grad()
            Risk.backward()
            optimizer.step()

        print(f'Risk: {Risk:.4f}, epoch: {i}')
        log_file.write(f'Risk: {Risk}, epoch: {i}\n')
        risk_list.append(Risk.detach().to('cpu'))
        KL_list.append(KL.detach().to('cpu'))
        RR_list.append(RR.detach().to('cpu'))
        with torch.no_grad():
            if i%5 == 0:
                model.eval()
                NMSE, Risk,output_stats = ev.eval_val(CSI,model_type,setup,model, dataloader_val,cov_type, lamba, device, dir_path)
                NMSE_estimation,mean_frob,mean_mu_signal_energy,Cov_part_LMMSE_energy,NMSE_only_mu = ev.channel_estimation(CSI,model, dataloader_val, sig_n,cov_type, dir_path, device)
                if model_type == 'Trajectory':
                    TPR1, TPR2 = ev.computing_MMD(CSI,setup, model, n_iterations, n_permutations, normed,bs_mmd, dataset_val, snapshots, dir_path,device)
                    eval_TPR1.append(TPR1)
                    eval_TPR2.append(TPR2)
                else:
                    TPR1 = 0
                    TPR2 = 0
                    eval_TPR1.append(TPR1)
                    eval_TPR2.append(TPR2)
                eval_risk.append(Risk.detach().to('cpu'))
                eval_NMSE.append(NMSE)
                eval_NMSE_estimation.append(NMSE_estimation)
                model.train()
                print(f'Evaluation - NMSE_prediction: {NMSE:.4f}, NMSE_estimation: {NMSE_estimation:.4f}, TPR1: {TPR1:.4f}, TPR2: {TPR2:.4f}, Risk: {Risk:.4f}')
                print(f'mean_frob: {mean_frob:.3f},mean_mu_signal_energy: {mean_mu_signal_energy:.3f},Cov_part_LMMSE_energy: {Cov_part_LMMSE_energy:.3f},NMSE_only_mu: {NMSE_only_mu:.3f}')
                log_file.write(f'Evaluation - NMSE_prediction: {NMSE:.4f}, NMSE_estimation: {NMSE_estimation:.4f}, TPR1: {TPR1:.4f}, TPR2: {TPR2:.4f} ,Risk: {Risk:.4f}\n')
                log_file.write(f'mean_frob: {mean_frob:.3f},mean_mu_signal_energy: {mean_mu_signal_energy:.3f},Cov_part_LMMSE_energy: {Cov_part_LMMSE_energy:.3f},NMSE_only_mu: {NMSE_only_mu:.3f}\n')
                if (i > 40) & (lr_adaption == False):
                    x_range_lr = torch.arange(5)
                    x_lr = torch.ones(5, 2)
                    x_lr[:, 0] = x_range_lr
                    beta_lr = torch.linalg.inv(x_lr.T @ x_lr) @ x_lr.T @ torch.tensor(eval_risk[-5:])[:, None]
                    slope_lr = beta_lr[0]
                    print('slope lr')
                    print(slope_lr)
                    log_file.write(f'slope of Evaluation ELBO (for learning rate): {slope_lr}\n')
                    if slope_lr > 0:
                        print('LEARNING RATE IS ADAPTED!')
                        log_file.write(f'LEARNING RATE IS ADAPTED!\n')
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/5
                        lr_adaption = True

                if (i > 200) & (lr_adaption == True):
                    x_range = torch.arange(15)
                    x = torch.ones(15, 2)
                    x[:, 0] = x_range
                    beta = torch.linalg.inv(x.T @ x) @ x.T @ torch.tensor(eval_risk[-15:])[:, None]
                    slope = beta[0]
                    print('slope')
                    print(slope)
                    log_file.write(f'slope of Evaluation ELBO: {slope}\n')

            if slope > 0:
                log_file.write('BREAKING CONDITION, slope positive\n')
                log_file.write(f'number epochs: {i}')
                break

    return risk_list,KL_list,RR_list,eval_risk,eval_NMSE, eval_NMSE_estimation, eval_TPR1,eval_TPR2