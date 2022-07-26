import math
import torch
import training as tr
import numpy as np
import matplotlib.pyplot as plt

def eval_val(GLOBAL_ARCHITECTURE, iteration, model,dataloader_val,risk_type, lamba,device, dir_path):

    iterator = iter(dataloader_val)
    samples = iterator.next()
    sample = samples.to(device)

    if (risk_type == 'kalmanVAE_toeplitz'):
        mu_out, B_out, C_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_toeplitz(sample, z, log_var, mu_out, B_out, C_out, mu_prior,logpre_prior, eps)

    if (risk_type == 'kalmanVAE_toeplitz_free_bits'):
        mu_out, B_out, C_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_toeplitz_free_bits(lamba, sample, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)

    if (risk_type == 'kalmanVAE_diagonal'):
        mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_diagonal(sample, z, log_var, mu_out, logpre_out, mu_prior,logpre_prior, eps)

    if (risk_type == 'kalmanVAE_diagonal_free_bits'):
        mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)


    if (risk_type == 'kMemoryHiddenMarkovVAE_diagonal'):
        mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_diagonal(sample, z, log_var, mu_out, logpre_out, mu_prior,logpre_prior, eps)

    if (risk_type == 'causal_kMemoryHMVAE_toeplitz_free_bits') | (risk_type == 'causal_kMemoryHMVAE_small_toeplitz_free_bits'):
        mu_out, B_out, C_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_toeplitz_free_bits(lamba, sample, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)

    if (risk_type == 'causal_kMemoryHMVAE_diagonal_free_bits'):
        mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)

    if risk_type == 'ApproxKMemoryHiddenMarkovVAE_diagonal':
        mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_diagonal(sample, z, log_var, mu_out, logpre_out, mu_prior,logpre_prior, eps)


    if risk_type == 'ApproxKMemoryHiddenMarkovVAE_diagonal_free_bits':
        mu_out, logpre_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_diagonal_free_bits(lamba, sample, z, log_var, mu_out, logpre_out,mu_prior, logpre_prior, eps)

    NMSE = channel_prediction(GLOBAL_ARCHITECTURE,model,dataloader_val,16,iteration,dir_path,device,'evaluation')
    return NMSE, Risk


def channel_prediction(GLOBAL_ARCHITECTURE,model,dataloader_val,knowledge,iteration,dir_path,device,PHASE):

    NMSE_list = []
    for ind,sample in enumerate(dataloader_val):
        samples = sample.to(device)
        time_stamps_per_unit = iteration[2]
        n_units = int(iteration[1][2]/time_stamps_per_unit)

        if GLOBAL_ARCHITECTURE == 'kalmanVAE':
            # encoding
            idx_upper_limit_encoder = int(math.floor(knowledge/time_stamps_per_unit))
            z_init = torch.ones(samples.size(0),iteration[0][0]).to(device)
            z_inf = torch.zeros(samples.size(0),iteration[0][0],idx_upper_limit_encoder).to(device)
            mu_z = model.encoder[0](samples[:,:,:,:time_stamps_per_unit],z_init)[0]
            z_inf[:,:,0] = mu_z
            if len(model.encoder) > 1:
                for idx in range(1,idx_upper_limit_encoder):
                    z_input = z_inf[:,:,idx-1].clone()
                    z_local = model.encoder[idx](samples[:,:,:,idx * time_stamps_per_unit: (idx + 1) * time_stamps_per_unit],z_input)[0]
                    z_inf[:,:,idx] = z_local
            if len(model.encoder) == 1:
                for idx in range(1, idx_upper_limit_encoder):
                    z_input = z_inf[:, :, idx - 1].clone()
                    z_local = model.encoder[0](samples[:, :, :, idx * time_stamps_per_unit: (idx + 1) * time_stamps_per_unit],z_input)[0]
                    z_inf[:, :, idx] = z_local

            # prior
            idx_first_prior = idx_upper_limit_encoder
            idx_first_dec = idx_first_prior
            z_list = torch.zeros(samples.size(0),iteration[0][0],n_units-idx_first_prior).to(device)
            z_input = z_inf[:,:,-1].clone()
            if len(model.prior_model) > 1:
                for idx in range(idx_first_prior,n_units):
                    z_local = model.prior_model[idx](z_input)[0]
                    z_list[:,:,idx-idx_first_prior] = z_local
                    z_input = z_local.clone()
            if len(model.prior_model) == 1:
                for idx in range(idx_first_prior,n_units):
                    z_local = model.prior_model[0](z_input)[0]
                    z_list[:,:,idx-idx_first_prior] = z_local
                    z_input = z_local.clone()
            # prediction
            x_list = torch.zeros(samples.size(0),iteration[1][0],iteration[1][1],(n_units-idx_first_dec) * time_stamps_per_unit).to(device)
            if len(model.decoder) > 1:
                for idx in range(idx_first_dec,n_units):
                    x_local = model.decoder[idx](z_list[:,:,idx - idx_first_prior])[0]
                    x_list[:,:,:,(idx-idx_first_dec)*time_stamps_per_unit:(idx-idx_first_dec+1)*time_stamps_per_unit] = x_local

            if len(model.decoder) == 1:
                for idx in range(idx_first_dec,n_units):
                    x_local = model.decoder[0](z_list[:,:,idx - idx_first_prior])[0]
                    x_list[:,:,:,(idx-idx_first_dec)*time_stamps_per_unit:(idx-idx_first_dec+1)*time_stamps_per_unit] = x_local

            predicted_samples = samples[:,:,:,idx_upper_limit_encoder*time_stamps_per_unit:]
            complete_x_list = torch.cat((samples[:,:,:,:idx_upper_limit_encoder*time_stamps_per_unit],x_list),dim=3)

            NMSE_list.append(torch.mean(torch.sum((predicted_samples - x_list) ** 2,dim=(1,2,3))/torch.sum(predicted_samples**2,dim=(1,2,3))).detach().to('cpu'))


        if (GLOBAL_ARCHITECTURE == 'causal_kMemoryHMVAE') | (GLOBAL_ARCHITECTURE == 'causal_kMemoryHMVAE_small'):
            memory = iteration[3]
            # encoding
            z_init = torch.ones(samples.size(0), iteration[0][0]).to(device)
            z_inf = torch.zeros(samples.size(0), iteration[0][0], math.floor(knowledge/time_stamps_per_unit)).to(device)
            x_init = torch.ones(samples.size(0),2,iteration[1][1],iteration[3]).to(device)
            x_input = torch.cat((x_init,samples[:,:,:,0][:,:,:,None]),dim=3)
            z_0 = model.encoder[0](x_input, z_init)[0]
            z_inf[:, :, 0] = z_0

            for idx in range(1,iteration[3]):
                z_input = z_inf[:,:,idx-1].clone()
                x_input = torch.cat((x_init[:,:,:,:iteration[3]-idx], samples[:, :, :, :idx+1]), dim=3)
                z_local = model.encoder[idx](x_input,z_input)[0]
                z_inf[:, :, idx] = z_local

            for idx in range(iteration[3], math.floor(knowledge/time_stamps_per_unit)):
                z_input = z_inf[:, :, idx - 1].clone()
                z_local = model.encoder[idx](samples[:, :, :, (idx - iteration[3]) * time_stamps_per_unit: (idx + 1) * time_stamps_per_unit],z_input)[0]
                z_inf[:, :, idx] = z_local

            # prior
            z_list = torch.zeros(samples.size(0), iteration[0][0], n_units - math.floor(knowledge/time_stamps_per_unit)).to(device)
            z_input = z_inf[:, :, -1].clone()
            if len(model.prior_model) > 1:
                for idx in range(math.floor(knowledge/time_stamps_per_unit), n_units):
                    z_local = model.prior_model[idx](z_input)[0]
                    z_list[:, :, idx - math.floor(knowledge/time_stamps_per_unit)] = z_local
                    z_input = z_local.clone()
            if len(model.prior_model) == 1:
                for idx in range(idx_first_prior, n_units):
                    z_local = model.prior_model[0](z_input)[0]
                    z_list[:, :, idx - idx_first_prior] = z_local
                    z_input = z_local.clone()
            # prediction
            z_total = torch.cat((z_inf,z_list),dim=2)
            x_list = torch.zeros(samples.size(0), iteration[1][0], iteration[1][1],(n_units - int(math.floor(knowledge / time_stamps_per_unit))) * time_stamps_per_unit).to(device)

            for idx in range(knowledge,n_units):
                z_input = z_total[:,:,idx-memory:idx+1]
                x_local = model.decoder[idx](z_input)[0]
                x_list[:, :, :, (idx-knowledge) :(idx-knowledge + 1)] = x_local

            predicted_samples = samples[:, :, :,  int(math.floor(knowledge / time_stamps_per_unit)) * time_stamps_per_unit:]
            complete_x_list = torch.cat((samples[:, :, :, :int(math.floor(knowledge / time_stamps_per_unit)) * time_stamps_per_unit], x_list), dim=3)
            NMSE_list.append(torch.mean(torch.sum((predicted_samples - x_list) ** 2, dim=(1, 2, 3)) / torch.sum(predicted_samples ** 2,dim=(1, 2, 3))).detach().to('cpu'))

    if PHASE == 'testing':
        prediction_visualization(samples,complete_x_list,dir_path)
    NMSE = np.mean(np.array(NMSE_list))
    return NMSE


def prediction_visualization(samples,complete_x_list,dir_path):
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