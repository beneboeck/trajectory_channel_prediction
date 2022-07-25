import math
import torch
import evaluation as ev

def risk_kalman_VAE_toeplitz_free_bits(lamba,x,z,log_var,mu_out,B,C,mu_prior,logpre_prior,eps):
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

def risk_kalman_VAE_diagonal_free_bits(lamba,x,z,log_var,mu_out,log_pre_out,mu_prior,logpre_prior,eps):
    x_compl = torch.complex(x[:,0,:,:],x[:,1,:,:])
    mu_compl = torch.complex(mu_out[:,0,:,:],mu_out[:,1,:,:])
    RR = torch.mean( torch.sum( - log_pre_out + log_pre_out.exp() * torch.abs(x_compl - mu_compl)**2 ,dim=(1,2)))
    #IR_term = - 0.5 * eps**2 - 0.5 * log_var
    IR_term = -0.5 * (log_var + 1)
    PR_term = 0.5 * (- logpre_prior + logpre_prior.exp() * (z - mu_prior)**2)
    KL = torch.mean(torch.sum(torch.max(lamba,IR_term + PR_term),dim=(1,2)))

    return RR + KL,RR,KL


def training_gen_NN(GLOBAL_ARCHITECTURE, iteration, lr, model, loader,dataloader_val, epochs, risk_type, lamba,
                      device, log_file,dir_path, dataset):

    risk_list= []
    KL_list = []
    RR_list = []
    eval_risk = []
    eval_NMSE = []
    slope = -1.

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    print('Start Training ')
    log_file.write('\n\nStart Training\n')

    for i in range(epochs):
        print('epoch')
        print(i)
        for ind, sample in enumerate(loader):
            sample = sample
            sample = sample.to(device)

            if (risk_type == 'kalmanVAE_toeplitz'):
                mu_out, B_out, C_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_toeplitz(sample, z, log_var, mu_out, B_out, C_out, mu_prior,logpre_prior, eps)

            if (risk_type == 'kalmanVAE_toeplitz_free_bits'):
                mu_out, B_out, C_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_toeplitz_free_bits(lamba, sample, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)

            if (risk_type == 'kalmanVAE_diagonal'):
                mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_diagonal(sample, z, log_var, mu_out, logpre_out, mu_prior,logpre_prior, eps)

            if (risk_type == 'kalmanVAE_diagonal_free_bits'):
                mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)


            if (risk_type == 'kMemoryHiddenMarkovVAE_diagonal'):
                mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_diagonal(sample, z, log_var, mu_out, logpre_out, mu_prior,logpre_prior, eps)

            if risk_type == 'causal_kMemoryHMVAE_toeplitz_free_bits':
                mu_out, B_out, C_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_toeplitz_free_bits(lamba, sample, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)

            if (risk_type == 'causal_kMemoryHMVAE_diagonal_free_bits'):
                mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)

            if risk_type == 'ApproxKMemoryHiddenMarkovVAE_diagonal':
                mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_diagonal(sample, z, log_var, mu_out, logpre_out, mu_prior,logpre_prior, eps)


            if risk_type == 'ApproxKMemoryHiddenMarkovVAE_diagonal_free_bits':
                mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
                mu_prior, logpre_prior = model.feed_prior(z)
                Risk, RR, KL = risk_kalman_VAE_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)

            optimizer.zero_grad()
            Risk.backward()
            optimizer.step()

        print(f'Risk: {Risk:.4f}, epoch: {i}')
        log_file.write(f'Risk: {Risk}, epoch: {i}\n')
        risk_list.append(Risk.detach().to('cpu'))
        KL_list.append(KL.detach().to('cpu'))
        RR_list.append(RR.detach().to('cpu'))
        with torch.no_grad():
            model.eval()
            NMSE, Risk = ev.eval_val(GLOBAL_ARCHITECTURE, iteration, model, dataloader_val, risk_type, lamba, device, dir_path)
            eval_risk.append(Risk.detach().to('cpu'))
            eval_NMSE.append(NMSE)
            print(f'Evaluation - NMSE: {NMSE:.4f},Risk: {Risk:.4f}')
            log_file.write(f'Evaluation - NMSE: {NMSE},Risk: {Risk}\n')
            if i > 30:
                x_range = torch.arange(20)
                x = torch.ones(20, 2)
                x[:, 0] = x_range
                beta = torch.linalg.inv(x.T @ x) @ x.T @ torch.tensor(eval_risk[-20:])[:, None]
                slope = beta[0]
                print('slope')
                print(slope)
                log_file.write(f'slope of Evaluation ELBO: {slope}\n')

        if slope > 0:
            log_file.write('BREAKING CONDITION, slope positive\n')
            log_file.write(f'number epochs: {i}')
            break

    return risk_list,KL_list,RR_list,eval_risk,eval_NMSE