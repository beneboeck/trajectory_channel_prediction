"""Module to create a multi path channel model.
Classes:
        SCMMulti: Class to build a multi path channel model.
"""
import numpy as np
import scm_helper_MIMO as scm_helper
from utils import *
def compute_polynom(x1,x2,coeff):
    return coeff[0]*x2+coeff[1]*x2**2+coeff[2]*x2**3+coeff[3]*x2**4+coeff[4]*x2**5+coeff[5]*x2**6+coeff[6]*x2**7+coeff[7]*x2**8+coeff[8]*x2**9+coeff[9]*x2**10+coeff[10]*x2**11+coeff[11]*x1+coeff[12]*x1*x2+coeff[13]*x1*x2**2+coeff[14]*x1*x2**3+coeff[15]*x1*x2**4+coeff[16]*x1*x2**5+coeff[17]*x1*x2**6+coeff[18]*x1*x2**7+coeff[19]*x1*x2**8+coeff[20]*x1*x2**9+coeff[21]*x1*x2**10+coeff[22]*x1*x2**11+coeff[23]*x1**2+coeff[24]*x1**2.*x2+coeff[25]*x1**2.*x2**2+coeff[26]*x1**2.*x2**3+coeff[27]*x1**2.*x2**4+coeff[28]*x1**2.*x2**5+coeff[29]*x1**2.*x2**6+coeff[30]*x1**2.*x2**7+coeff[31]*x1**2.*x2**8+coeff[32]*x1**2.*x2**9+coeff[33]*x1**2.*x2**10+coeff[34]*x1**3+coeff[35]*x1**3.*x2+coeff[36]*x1**3.*x2**2+coeff[37]*x1**3.*x2**3+coeff[38]*x1**3.*x2**4+coeff[39]*x1**3.*x2**5+coeff[40]*x1**3.*x2**6+coeff[41]*x1**3.*x2**7+coeff[42]*x1**3.*x2**8+coeff[43]*x1**3.*x2**9+coeff[44]*x1**4+coeff[45]*x1**4.*x2+coeff[46]*x1**4.*x2**2+coeff[47]*x1**4.*x2**3+coeff[48]*x1**4.*x2**4+coeff[49]*x1**4.*x2**5+coeff[50]*x1**4.*x2**6+coeff[51]*x1**4.*x2**7+coeff[52]*x1**4.*x2**8+coeff[53]*x1**5+coeff[54]*x1**5.*x2+coeff[55]*x1**5.*x2**2+coeff[56]*x1**5.*x2**3+coeff[57]*x1**5.*x2**4+coeff[58]*x1**5.*x2**5+coeff[59]*x1**5.*x2**6+coeff[60]*x1**5.*x2**7+coeff[61]*x1**6+coeff[62]*x1**6.*x2+coeff[63]*x1**6.*x2**2+coeff[64]*x1**6.*x2**3+coeff[65]*x1**6.*x2**4+coeff[66]*x1**6.*x2**5+coeff[67]*x1**6.*x2**6+coeff[68]*x1**7+coeff[69]*x1**7.*x2+coeff[70]*x1**7.*x2**2+coeff[71]*x1**7.*x2**3+coeff[72]*x1**7.*x2**4+coeff[73]*x1**7.*x2**5+coeff[74]*x1**8+coeff[75]*x1**8.*x2+coeff[76]*x1**8.*x2**2+coeff[77]*x1**8.*x2**3+coeff[78]*x1**8.*x2**4+coeff[79]*x1**9+coeff[80]*x1**9.*x20+coeff[81]*x1**9.*x2**2+coeff[82]*x1**9.*x2**3+coeff[83]*x1**10+coeff[84]*x1**10.*x2+coeff[85]*x1**10.*x2**2+coeff[86]*x1**11+coeff[87]*x1**11.*x2+coeff[88]*1+coeff[89]*x1**12+coeff[90]*x2**12



class SCMMulti:
    """Class to build a multi path channel model.
    This class defines a multi path channel model.
    Public Methods:
    Instance Variables:
    """

    def __init__(self,path_sigma_BS=2.0, path_sigma_MS=35.0, n_path=3,coords_list = None,sigma_grid_list = None, convex_sigmas = None):
        """Initialize multi path channel model.
        First, initialise all variables belonging to the multi path channel model.
        """
        self.path_sigma_BS = path_sigma_BS
        self.path_sigma_MS = path_sigma_MS
        self.n_path = n_path

        #interpolator_list, # double list with first index: paths, second index: latent variable (2 * number_paths) first gains, then DoAs
        #convex_sigmas, # np array with length paths
        if (coords_list is None) & (sigma_grid_list is None) & (convex_sigmas is None):
            interpolator_list = []
            sigma_grid_list = []
            coords_list = []

            for n in range(self.n_path):
                interpolator_list.append([])
                sigma_grid_list.append([])
                coords_list.append([])
                for a in range(2 * self.n_path):
                    interpolator,sigma_grid_samples,coords = compute_interpolator(sigma_grid_list = None,coords=None)
                    interpolator_list[n].append(interpolator)
                    sigma_grid_list[n].append(sigma_grid_samples)
                    coords_list[n].append(coords)
    
            convex_sigmas = np.random.rand(self.n_path)
            convex_sigmas = convex_sigmas / np.sum(convex_sigmas)
        else:
            interpolator_list = []
            for n in range(self.n_path):
                interpolator_list.append([])
                for a in range(2 * self.n_path):
                    interpolator, sigma_grid_samples, coords = compute_interpolator(sigma_grid_list=sigma_grid_list[n,a,:,:], coords=coords_list[n,a,:,:])
                    interpolator_list[n].append(interpolator)

        self.convex_sigmas = convex_sigmas
        self.interpolator_list = interpolator_list
        self.sigma_grid_list = np.array(sigma_grid_list)
        self.coords_list = np.array(coords_list)

    def save_parameters(self,n_paths):
        np.save('convex_sigmas_npaths_' + str(n_paths),self.convex_sigmas)
        np.save('sigma_grid_samples_npaths_'+ str(n_paths),self.sigma_grid_list)
        np.save('coords_samples_npaths_'+ str(n_paths),self.coords_list)

    def generate_channel(
        self,
        n_batches,
        n_snapshots,
        n_coherence,
        n_antennas_BS,
        n_antennas_MS,
        rng=np.random.random.__self__
    ):
        """Generate multi path model parameters.
        Function that generates the multi path model parameters for given inputs.
        """

        # generate the input dependent standard deviations

        h = np.zeros([n_batches,n_coherence, n_snapshots, n_antennas_BS*n_antennas_MS], dtype=np.complex64)
        t_BS = np.zeros([n_batches, n_snapshots,n_antennas_BS], dtype=np.complex64)
        t_MS = np.zeros([n_batches, n_snapshots,n_antennas_MS], dtype=np.complex64)

        angles_BS_list = np.zeros((n_batches,n_snapshots,self.n_path))
        gains_list = np.zeros((n_batches,n_snapshots,self.n_path))

        #angles = np.linspace(-90,90,n_batches)
        for i in range(n_batches):
            if i % 1000 == 0:
                print('i')
                print(i)
            gains_sample = np.zeros((n_snapshots,self.n_path))
            angles_BS_sample = np.zeros((n_snapshots,self.n_path))
            gains = rng.rand(self.n_path)
            gains = gains / np.sum(gains, axis=0)
            angles_BS = (rng.rand(self.n_path) - 0.5) * 2 * np.pi
            angles_MS = (rng.rand(self.n_path) - 0.5) * 2 * np.pi
            gains_sample[0, :] = gains
            angles_BS_sample[0, :] = angles_BS
            h[i,:, 0, :], t_BS[i, 0, :], t_MS[i, 0, :] = scm_helper.chan_from_spectrum(n_coherence, n_antennas_BS,n_antennas_MS,angles_BS_sample[0, :], angles_MS,gains_sample[0, :],self.path_sigma_BS,self.path_sigma_MS, rng=rng)

            for s in range(1,n_snapshots):
                
                sigma_gains = np.zeros(self.n_path)
                sigma_DoAs = np.zeros(self.n_path)
                for m in range(self.n_path):
                    for n in range(self.n_path):
                        input = np.array([gains_sample[s-1,m],angles_BS_sample[s-1,m]])
                        sigma_gains[m] = sigma_gains[m] + self.convex_sigmas[n] * self.interpolator_list[n][m](input[None,:])
                        sigma_DoAs[m] = sigma_DoAs[m] + self.convex_sigmas[n] * self.interpolator_list[n][m + self.n_path](input[None,:])
                        if self.interpolator_list[n][m + self.n_path](input[None,:]) < 0:
                            print('input')
                            print(input)
                            raise ValueError
                gains_sample[s,:] = gains_sample[s-1,:] + 0.75 * sigma_gains * rng.randn(self.n_path)
                gains_sample[gains_sample > 1] = 0.99
                gains_sample[gains_sample < 0] = 0.01
                #gains_sample[s,:] = gains_sample[s,:]/np.sum(gains_sample[s,:])
                angles_BS_sample[s,:] = angles_BS_sample[s-1,:] + sigma_DoAs * rng.randn(self.n_path)
                angles_BS_sample[angles_BS_sample > np.pi] = np.pi
                angles_BS_sample[angles_BS_sample < -np.pi] = -np.pi
                h[i,:,s,:], t_BS[i,s, :], t_MS[i,s, :] = scm_helper.chan_from_spectrum(n_coherence, n_antennas_BS,n_antennas_MS, angles_BS_sample[0,:], angles_MS,gains_sample[0,:], self.path_sigma_BS,self.path_sigma_MS, rng=rng)
            angles_BS_list[i,:,:] = angles_BS_sample
            gains_list[i,:,:] = gains_sample
        return h, t_BS, t_MS, gains_list, angles_BS_list,self.interpolator_list

    def get_config(self):
        config = {
            'path_sigma_BS': self.path_sigma_BS,
            'path_sigma_MS': self.path_sigma_MS,
            'n_path': self.n_path
        }
        return config