import time
from functools import wraps
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import scipy
import math
import matplotlib.pyplot as plt


def apply_DFT(sample_set):
    # sample_set [N_SAMPLES,2,N_ANTENNAS,N_SNAPSHOTS]
    n_ant = sample_set.shape[2]
    F = np.zeros((n_ant,n_ant),dtype=np.cfloat)
    for m in range(n_ant):
        for n in range(n_ant):
            F[m,n] = 1/np.sqrt(n_ant) * np.exp(1j * 2 * math.pi * (m * n)/n_ant)

    sample_set_compl = sample_set[:,0,:,:] + 1j * sample_set[:,1,:,:]
    transformed_set = np.einsum('mn,knl -> kml',F,sample_set_compl)
    realed_set = np.zeros((sample_set.shape))
    realed_set[:,0,:,:] = np.real(transformed_set)
    realed_set[:,1,:,:] = np.imag(transformed_set)
    return realed_set

def apply_IDFT(sample_set):
    # sample_set [N_SAMPLES,2,N_ANTENNAS,N_SNAPSHOTS]
    sample_set = np.array(sample_set.detach().to('cpu'))
    n_ant = sample_set.shape[2]
    F = np.zeros((n_ant,n_ant),dtype=np.cfloat)
    for m in range(n_ant):
        for n in range(n_ant):
            F[m,n] = 1/np.sqrt(n_ant) * np.exp(-1j * 2 * math.pi * (m * n)/n_ant)

    sample_set_compl = sample_set[:,0,:,:] + 1j * sample_set[:,1,:,:]
    transformed_set = np.einsum('mn,knl -> kml',F,sample_set_compl)
    realed_set = np.zeros((sample_set.shape))
    realed_set[:,0,:,:] = np.real(transformed_set)
    realed_set[:,1,:,:] = np.imag(transformed_set)
    return realed_set

def network_architecture_search():
    LD = np.random.choice([16,24,32,40]).item()
    memory = np.random.choice(range(3,10)).item()
    rnn_bool = np.random.choice([False,True]).item()
    BN = np.random.choice([False]).item()
    en_layer = np.random.choice([2,3]).item()
    en_width = np.random.choice([4,6,8]).item()
    pr_layer = np.random.choice([2,3,4]).item()
    pr_width = np.random.choice([3,6,9]).item()
    de_layer = np.random.choice([4,5]).item()
    de_width = np.random.choice([6,8,12]).item()
    cov_type = np.random.choice(['Toeplitz','DFT']).item()
    cov_type = 'DFT'
    prepro = np.random.choice(['None','DFT']).item()
    if cov_type == 'Toeplitz':
        prepro = 'DFT'
    n_conv = np.random.choice([1,2]).item()
    cnn_bool = np.random.choice([True,False]).item()
    if cov_type == 'Toeplitz':
        cnn_bool = False
    LB_var_dec = round(np.random.uniform(low = 0.0001, high = 0.01),4)
    UB_var_dec = round(np.random.uniform(low=0.5, high = 1),4)

    return LD,memory,rnn_bool,en_layer,en_width,pr_layer,pr_width,de_layer,de_width,cov_type,BN,prepro,n_conv,cnn_bool,LB_var_dec,UB_var_dec

def network_architecture_search_VAE():
    LD = np.random.choice([16,24,32,40,48,56]).item()
    LD = 56
    #LD = np.random.choice([40, 48, 56]).item()
    conv_layer = np.random.choice([0,1,2,3]).item()
    conv_layer = 0
    total_layer = np.random.choice([3,4,5]).item()
    total_layer = 3
    out_channel = np.random.choice([128]).item()
    k_size = np.random.choice([7,9]).item()
    BN = np.random.choice([False,True]).item()
    BN = False
    cov_type = np.random.choice(['Toeplitz','DFT']).item()
    cov_type = 'DFT'
    prepro = np.random.choice(['None', 'DFT']).item()
    if cov_type == 'Toeplitz':
        prepro = 'None'
    LB_var_dec = round(np.random.uniform(low=0.0001, high=0.01), 4)
    UB_var_dec = round(np.random.uniform(low=0.5, high=1), 4)

    return LD,conv_layer,total_layer,out_channel,k_size,cov_type,prepro,LB_var_dec,UB_var_dec,BN

def network_architecture_search_TraVAE():
    LD = np.random.choice([4*16,16*8,16*16]).item()
    conv_layer = np.random.choice([0,1,2,3]).item()
    total_layer = np.random.choice([3,4,5]).item()
    out_channel = np.random.choice([64,128]).item()
    k_size = np.random.choice([7,9]).item()
    cov_type = np.random.choice(['Toeplitz','DFT']).item()
    cov_type = 'DFT'
    prepro = np.random.choice(['None']).item()
    LB_var_dec = round(np.random.uniform(low=0.0001, high=0.01), 4)
    UB_var_dec = round(np.random.uniform(low=0.5, high=1), 4)
    BN = np.random.choice([False, True]).item()
    BN = False
    return LD,conv_layer,total_layer,out_channel,k_size,cov_type,prepro,LB_var_dec,UB_var_dec,BN

def save_risk(risk_list,RR_list,KL_list,model_path,title):
    risk = np.array(risk_list)
    np.save(model_path + '/risk_numpy',risk)
    plt.plot(risk,linewidth=1,label = 'Risk')
    plt.plot(np.array(RR_list), linewidth=1, label = 'RR')
    plt.plot(np.array(KL_list), linewidth=1, label = 'KL')
    plt.title(title)
    plt.legend()
    plt.savefig(model_path + '/' + title,dpi = 300)
    plt.close()

def save_risk_single(risk_list,model_path,title):
    risk = np.array(risk_list)
    np.save(model_path + '/risk_numpy',risk)
    plt.plot(risk,linewidth=1)
    plt.title(title)
    plt.savefig(model_path + '/' + title,dpi = 300)
    plt.close()

def compute_interpolator(sigma_grid_list = None,coords=None):
    if (sigma_grid_list is None) & (coords is None):
        DoA_grid = np.linspace(-np.pi, np.pi, 4)
        PL_grid = np.linspace(0, 1, 4)
        features_grid = []
        features_grid.append(PL_grid)
        features_grid.append(DoA_grid)
        coords = np.meshgrid(*features_grid,copy=False)
        coords_list = []
        for n in range(2):
            coords_list.append(coords[n].reshape(-1, 1))
        coords_list = np.squeeze(np.array(coords_list)).T
        sigma_grid_samples = 0.05 + 0.15 * np.random.rand(4 ** 2, 1)
    else:
        sigma_grid_samples = sigma_grid_list
        coords_list = coords

    interpolator = scipy.interpolate.RBFInterpolator(coords_list, sigma_grid_samples, kernel='cubic')
    
    return interpolator,sigma_grid_samples,coords_list

def num_combinations(max_degree,n_paths):

    n_variables = n_paths * 2
    number = 0
    for i in range(max_degree+1):
        number = number + int(scipy.special.binom(n_variables + i - 1,i))
    return number


def generating_clf_list(number_antennas,number_coeff,n_paths,max_degree):
    sigma_grid_list = []
    for a in range(number_antennas):
        if n_paths > 2:
            sigma_grid_samples = 0.1 + 0.4 * np.random.rand(10**(2*n_paths), 1)
        else:
            sigma_grid_samples = 0.1 + 0.4 * np.random.rand(((2 * n_paths) ** 2 * np.ceil(np.sqrt(number_coeff)) ** (2 * n_paths)).astype(int), 1)
        sigma_grid_list.append(sigma_grid_samples)
    if n_paths > 2:
        DoA_grid = np.linspace(-np.pi, np.pi,10)
        PL_grid = np.linspace(0, 1,10)
    else:
        DoA_grid = np.linspace(-np.pi, np.pi, (2 * np.ceil(np.sqrt(number_coeff))))
        PL_grid = np.linspace(0, 1, (2 * np.ceil(np.sqrt(number_coeff))))

    print('first for loop')
    features_grid = []
    for n in range(n_paths):
        features_grid.append(DoA_grid)
        features_grid.append(PL_grid)

    coords = np.meshgrid(*features_grid)
    print('second for loop')
    coords_list = []
    for n in range(2 * n_paths):
        coords_list.append(coords[n].reshape(-1, 1))
    print('third for loop')
    coords_list = np.squeeze(np.array(coords_list)).T
    print(coords_list.shape)
    print(sigma_grid_samples.shape)
    poly = PolynomialFeatures(degree=max_degree)
    print('test')
    X_ = poly.fit_transform(coords_list)
    print('test2')
    print(sigma_grid_samples.shape)
    clf_list = []
    for a in range(number_antennas):
        print(a)
        clf = linear_model.LinearRegression()
        clf.fit(X_, sigma_grid_list[a])
        clf_list.append(clf)

    print('clf generation is done')
    return poly,clf_list



def crandn(*arg, rng=np.random.random.__self__):
    #np.random.seed()
    return np.sqrt(0.5) * (rng.randn(*arg) + 1j * rng.randn(*arg))


def timethis(func):
    """A decorator that prints the execution time.
    Example:
        Write @utils.timethis before a function definition:
        @utils.timthis
        def my_function():
            pass
        Then, every time my_function is called, the execution time is printed.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        # hours
        h = (toc - tic) // (60 * 60)
        s = (toc - tic) % (60 * 60)
        print(
            'elapsed time of {}(): '
            '{:.0f} hour(s) | {:.0f} minute(s) | {:.5f} second(s).'
            .format(func.__name__, h, s // 60, s % 60)
        )
        return result
    return wrapper