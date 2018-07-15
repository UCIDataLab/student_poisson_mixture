import numpy as np
from scipy.misc import factorial


def loggamma_vec(x_vec, a, b):
    """
    sum_i log Gamma(x_vec[i], a[i], b[i])

    Parameters
    ----------
    x_vec : np.array
    a : np.array or float
    b : np.array or float

    Returns
    -------
    float
    """
    p_x = a * np.log(b) - np.log(factorial(a - 1.0)) + (a - 1.0) * np.log(x_vec) - b * x_vec
    return np.sum(p_x)


def logpoisson_vec(lam_vec, x_vec):
    """
    sum_i logPoisson(lam_vec[i], x_vec[i])

    Parameters
    ----------
    lam_vec : np.array
        1 dimensional (D,) size numpy array for the Poisson mean
    x_vec : np.array
        1 dimensional (D,) size numpy array

    Returns
    -------
    float

    """
    lam_vec = np.array(lam_vec)
    if np.prod(lam_vec) == 0:
        for i, l in enumerate(lam_vec):
            if l == 0:
                lam_vec[i] = 1e-7
    x_vec = np.array(x_vec)
    return np.sum(-lam_vec + x_vec * np.log(lam_vec) - np.log(factorial(x_vec)))


def logpoisson_vec_all(lam_vec, x_mat, gam_prior=False, a=1.0, b=1.0):
    """
    Get an array of log likelihood for each x_mat[i], using lam_vec
    loglik[i] = logpoisson_vec(lam_vec, x_mat[i])

    Parameters
    ----------
    lam_vec : np.array
        1 dimensional (D,) size numpy array for the Poisson mean
    x_mat : np.array
        2 dimensional (N, D) size numpy array

    Returns
    -------
    np.array
        (N,) size log probabilities
    """
    logprob= np.zeros(x_mat.shape[0])
    if gam_prior:
        for i in range(x_mat.shape[0]):
            xi = x_mat[i]
            const = np.log(factorial(a + xi - 1)) - np.log(factorial(a - 1)) \
                    - np \
                    .log ( factorial(xi)) + a* np.log(b) - (a + xi)* np.log(b+1)
            logprob[i] = loggamma_vec( lam_vec, a+xi, b+1) + np.sum(const)
    else:
        for i in range(x_mat.shape[0]):
            logprob[i] = logpoisson_vec(lam_vec, x_mat[i])
    return logprob

