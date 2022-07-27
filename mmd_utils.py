import torch


def Pdist2(x, y):
    """Compute the paired distance between x and y"""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))

    Pdist[Pdist < 0] = 0
    return Pdist


def eval_mmd2(Kx, Ky, Kxy, use_1sample_U=True):
    """Compute MMD given the kernel matrix"""
    # create the complete matrix
    Kxxy = torch.cat((Kx, Kxy), 1)
    Kyxy = torch.cat((Kxy.transpose(0, 1), Ky), 1)
    Kxyxy = torch.cat((Kxxy, Kyxy), 0)

    nx = Kx.shape[0]
    ny = Ky.shape[0]
    xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
    yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
    # one-sample U-statistic when nx=ny
    if use_1sample_U:
        xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
    else:
        xy = torch.div(torch.sum(Kxy), (nx * ny))
    mmd2 = xx - 2 * xy + yy

    return mmd2, Kxyxy


def eval_mmd2_with_variance(Kx, Ky, Kxy, use_1sample_U=True):
    """Compute MMD and variance of MMD given the kernel matrix"""
    # create the complete matrix
    Kxxy = torch.cat((Kx, Kxy), 1)
    Kyxy = torch.cat((Kxy.transpose(0, 1), Ky), 1)
    Kxyxy = torch.cat((Kxxy, Kyxy), 0)

    nx = Kx.shape[0]
    ny = Ky.shape[0]
    xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
    yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
    # one-sample U-statistic when nx=ny
    if use_1sample_U:
        xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
    else:
        xy = torch.div(torch.sum(Kxy), (nx * ny))
    mmd2 = xx - 2 * xy + yy

    # estimate variance
    hh = Kx + Ky - Kxy - Kxy.transpose(0, 1)

    V1 = torch.dot(hh.sum(1) / ny, hh.sum(1) / ny) / ny
    V2 = hh.sum() / nx / nx
    varEst = 4 * (V1 - V2 ** 2)
    if varEst == 0.0:
        print('error_var!!' + str(V1))

    return mmd2, Kxyxy, varEst


def MMDu(X, Y, sigma0):
    """Generate the kernel matrix for MMD computation"""
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)

    # we use a simple gaussian kernel
    Kx = torch.exp(-Dxx / sigma0)
    Ky = torch.exp(-Dyy / sigma0)
    Kxy = torch.exp(-Dxy / sigma0)

    return eval_mmd2_with_variance(Kx, Ky, Kxy, use_1sample_U=True)


""" *************************************************** mixture *************************************************** """


def MMDu_mixture(X, Y, sigma0_list):
    """Generate the kernel matrix for MMD computation"""
    Kx = 0.0
    Ky = 0.0
    Kxy = 0.0

    for jj in range(len(sigma0_list)):
        sigma0 = sigma0_list[jj]

        Dxx = Pdist2(X, X)
        Dyy = Pdist2(Y, Y)
        Dxy = Pdist2(X, Y)

        # we use a simple gaussian kernel
        Kx += torch.exp(-Dxx / sigma0)
        Ky += torch.exp(-Dyy / sigma0)
        Kxy += torch.exp(-Dxy / sigma0)

    return eval_mmd2_with_variance(Kx, Ky, Kxy, use_1sample_U=True)
