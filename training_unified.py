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

    return RR + KL,RR,KL

def risk_free_bits(lamba,x,mu,log_var,mu_out,Gamma):
    x = torch.complex(torch.squeeze(x[:,:,:32]), torch.squeeze(x[:,:,32:]))
    Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j
    M, pivots = torch.lu(Gamma)
    P, L, U = torch.lu_unpack(M, pivots)
    diagU = torch.diagonal(U, dim1=1, dim2=2)
    log_detGamma = torch.sum(torch.log(torch.abs(diagU)), dim=1)
    argument = torch.einsum('ij,ij->i', torch.conj(x - mu_out), torch.einsum('ijk,ik->ij', Gamma, x - mu_out))
    Rec_err = torch.real(torch.mean(- log_detGamma + argument))
    KL =  torch.mean(torch.sum(torch.max(lamba,-0.5 * (1 + log_var - mu ** 2 - (log_var).exp())),dim=1))
    return Rec_err + KL,Rec_err,KL

def training_gen_NN(model_type,setup,lr, cov_type,model, loader,dataloader_val, epochs, lamba,sig_n, device, log_file,dir_path,n_iterations, n_permutations, normed,bs_mmd, dataset_val, snapshots):

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
            if cov_type == 'DFT':
                sample = samples[2]
            else:
                sample = samples[0]
            sample = sample.to(device)

            if (model_type == 'Trajectory') & (cov_type == 'Toeplitz'):
                out, z, eps, mu_inf, log_var = model(sample)
                mu_out, B_out, C_out = out
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_toeplitz_free_bits(lamba, sample, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)

            if (model_type == 'Trajectory') & (cov_type == 'DFT'):
                out, z, eps, mu_inf, log_var = model(sample)
                mu_out, logpre_out = out
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)

            if (model_type == 'Single'):
                sample = sample[:,:,:,-1]
                print(sample.size())
                mu_out, Gamma, mu, log_var = model(sample)
                Risk, RR, KL = risk_free_bits(lamba,sample,mu,log_var,mu_out,Gamma)


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
                NMSE, Risk = ev.eval_val(setup,model, dataloader_val,cov_type, lamba, device, dir_path)
                NMSE_estimation = ev.channel_estimation(setup, model, dataloader_val, sig_n, dir_path, device)
                TPR1, TPR2 = ev.computing_MMD(setup, model, n_iterations, n_permutations, normed,bs_mmd, dataset_val, snapshots, dir_path,device)
                eval_risk.append(Risk.detach().to('cpu'))
                eval_NMSE.append(NMSE)
                eval_NMSE_estimation.append(NMSE_estimation)
                eval_TPR1.append(TPR1)
                eval_TPR2.append(TPR2)
                model.train()
                print(f'Evaluation - NMSE_prediction: {NMSE:.4f}, NMSE_estimation: {NMSE_estimation:.4f}, TPR1: {TPR1:.4f}, TPR2: {TPR2:.4f}, Risk: {Risk:.4f}')
                log_file.write(f'Evaluation - NMSE_prediction: {NMSE:.4f}, NMSE_estimation: {NMSE_estimation:.4f}, TPR1: {TPR1:.4f}, TPR2: {TPR2:.4f} ,Risk: {Risk:.4f}\n')
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

                if (i > 300) & (lr_adaption == True):
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