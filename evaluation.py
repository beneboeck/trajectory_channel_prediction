import math
import torch
import training as tr
import numpy as np

def eval_val(GLOBAL_ARCHITECTURE, iteration, model,dataloader_val,risk_type, lamba,device, log_file):

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

    if risk_type == 'kMemoryHiddenMarkovVAE_toeplitz_free_bits':
        mu_out, B_out, C_out, z, eps, mu_inf, log_var = model(sample)
        mu_prior, logpre_prior = model.feed_prior(z)
        Risk, RR, KL = tr.risk_kalman_VAE_toeplitz_free_bits(lamba, sample, z, log_var, mu_out, B_out,C_out, mu_prior, logpre_prior, eps)

    if (risk_type == 'kMemoryHiddenMarkovVAE_diagonal_free_bits'):
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

    NMSE = channel_prediction(GLOBAL_ARCHITECTURE,model,dataloader_val,16,iteration,device)
    return NMSE, Risk


def channel_prediction(GLOBAL_ARCHITECTURE,model,dataloader_val,knowledge,iteration,device):

    NMSE_list = []
    for ind,sample in enumerate(dataloader_val):
        samples = samples.to(device)
        time_stamps_per_unit = iteration[2]
        n_units = int(iteration[1][2]/time_stamps_per_unit)

        if GLOBAL_ARCHITECTURE == 'kalmanVAE':
            # encoding
            idx_upper_limit_encoder = int(math.floor(knowledge/time_stamps_per_unit))
            z_init = torch.ones(samples.size(0),iteration[0][0]).to(device)
            print('here')
            print(z_init.size())
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

            NMSE.append(torch.mean(torch.sum((predicted_samples - x_list) ** 2,dim=(1,2,3))/torch.sum(x_list**2,dim=(1,2,3))).detach().to('cpu'))


        if (GLOBAL_ARCHITECTURE == 'kMemoryHiddenMarkovVAE') | (GLOBAL_ARCHITECTURE == 'WN_kMemoryHiddenMarkovVAE'):
            memory = iteration[3]
            # encoding
            z_init = torch.ones(batchsize, iteration[0][0]).to(device)
            z_inf = torch.zeros(batchsize, iteration[0][0], math.floor(knowledge/time_stamps_per_unit)).to(device)
            z_0 = model.encoder[0](samples[:, :, :, :time_stamps_per_unit*(memory+1)], z_init)[0]
            z_inf[:, :, 0] = z_0
            if len(model.encoder) > 1:
                for idx in range(1, math.floor(knowledge/time_stamps_per_unit)):
                    z_input = z_inf[:, :, idx - 1].clone()
                    z_local = model.encoder[idx](samples[:, :, :, idx * time_stamps_per_unit: (idx + memory + 1) * time_stamps_per_unit],z_input)[0]
                    z_inf[:, :, idx] = z_local
            if len(model.encoder) == 1:
                for idx in range(1, math.floor(knowledge/time_stamps_per_unit)):
                    z_input = z_inf[:, :, idx - 1].clone()
                    z_local = model.encoder[0](samples[:, :, :, idx * time_stamps_per_unit: (idx + memory + 1) * time_stamps_per_unit],z_input)[0]
                    z_inf[:, :, idx] = z_local

            # prior
            z_list = torch.zeros(batchsize, iteration[0][0], n_units - math.floor(knowledge/time_stamps_per_unit)).to(device)
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
            x_list = torch.zeros(batchsize, iteration[1][0], iteration[1][1],(n_units - int(math.floor(knowledge / time_stamps_per_unit))) * time_stamps_per_unit).to(device)
            if len(model.decoder) > 1:
                print('ja')
                for idx in range(memory):
                    print(idx)
                    z_input = torch.cat((z_inf[:,:,-(memory - idx):],z_list[:,:,:idx+1]),dim=2)
                    x_local = model.decoder[math.floor(knowledge/time_stamps_per_unit) + idx](z_input)[0]
                    x_list[:,:,:,idx*time_stamps_per_unit:(idx+1) * time_stamps_per_unit] = x_local
                for idx in range(math.floor(knowledge/time_stamps_per_unit) + memory, n_units):
                    z_input = z_list[:,:,idx - (math.floor(knowledge/time_stamps_per_unit) + memory):(idx + memory + 1) - (math.floor(knowledge/time_stamps_per_unit) + memory)]
                    x_local = model.decoder[idx](z_input)[0]
                    x_list[:, :, :,(idx - math.floor(knowledge/time_stamps_per_unit)) * time_stamps_per_unit:(idx - math.floor(knowledge/time_stamps_per_unit) + 1) * time_stamps_per_unit] = x_local

            if len(model.decoder) == 1:
                for idx in range(memory):
                    z_input = torch.cat((z_inf[:,:,-(memory - idx)],z_list[:,:,:idx+1]),dim=2)
                    x_local = model.decoder[0](z_input)[0]
                    x_list[:,:,:,idx*time_stamps_per_unit:(idx+1) * time_stamps_per_unit] = x_local
                for idx in range(math.floor(knowledge/time_stamps_per_unit) + memory, n_units):
                    z_input = z_list[:,:,idx - (math.floor(knowledge/time_stamps_per_unit) + memory):(idx + memory + 1) - (math.floor(knowledge/time_stamps_per_unit) + memory)]
                    x_local = model.decoder[0](z_input)[0]
                    x_list[:, :, :,(idx - math.floor(knowledge/time_stamps_per_unit)) * time_stamps_per_unit:(idx - math.floor(knowledge/time_stamps_per_unit) + 1) * time_stamps_per_unit] = x_local

            predicted_samples = samples[:, :, :,  int(math.floor(knowledge / time_stamps_per_unit)) * time_stamps_per_unit:]
            print('joo')
            print(predicted_samples.size())
            print(x_list.size())
            MSE = torch.mean((predicted_samples - x_list) ** 2)
            complete_x_list = torch.cat((samples[:, :, :, :int(math.floor(knowledge / time_stamps_per_unit)) * time_stamps_per_unit], x_list), dim=3)


    # fig, ax = plt.subplots(4, 6, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(18, 4))
    #
    # for n in range(6):
    #     sample = samples[n, :, :, :]
    #     sample = sample[None, :, :, :]
    #
    #     mu_out = complete_x_list[n, :, :, :]
    #     mu_out = torch.squeeze(mu_out).to('cpu').detach()
    #     mu_out = torch.complex(mu_out[0, :, :], mu_out[1, :, :])
    #     abs_out = torch.abs(mu_out)
    #     angle_out = torch.angle(mu_out)
    #
    #     sample = sample.to('cpu')
    #     sample = torch.squeeze(sample)
    #     sample = torch.complex(sample[0, :, :], sample[1, :, :])
    #     abs_sample = torch.abs(sample)
    #     angle_sample = torch.angle(sample)
    #
    #     ax[int(np.floor(n / 3) * 2), int(n % 3 * 2)].imshow(abs_sample.numpy(), cmap='hot')
    #     ax[int(np.floor(n / 3) * 2), int(n % 3 * 2)].set_xticks([])
    #     ax[int(np.floor(n / 3) * 2), int(n % 3 * 2)].set_yticks([])
    #
    #     ax[int(np.floor(n / 3) * 2), int(n % 3 * 2) + 1].imshow(abs_out.numpy(), cmap='hot')
    #     ax[int(np.floor(n / 3) * 2), int(n % 3 * 2) + 1].set_xticks([])
    #     ax[int(np.floor(n / 3) * 2), int(n % 3 * 2) + 1].set_yticks([])
    #
    #     ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2)].imshow(angle_sample.numpy(), cmap='hot')
    #     ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2)].set_xticks([])
    #     ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2)].set_yticks([])
    #
    #     ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2 + 1)].imshow(angle_out.numpy(), cmap='hot')
    #     ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2 + 1)].set_xticks([])
    #     ax[int(np.floor(n / 3) * 2) + 1, int(n % 3 * 2 + 1)].set_yticks([])
    #
    # fig.suptitle('Real Domain (antennas)- NW:original abs,NE:estimated abs,SW:original phase,SE:estimated phase')
    #
    # fig.savefig(dir_path + '/heat_map_for_prediction_' + time + '_' + str(iteration) + '_' + scenario + 'real.png', dpi=300)
    # plt.close('all')
    NMSE = np.mean(np.array(NMSE_list))
    return NMSE
