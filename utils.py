import time
from functools import wraps
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import scipy


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