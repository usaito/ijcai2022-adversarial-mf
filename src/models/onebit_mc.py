"""1-bit MC algorithm.

references:
- paper
https://arxiv.org/abs/1910.12774

- implementations
https://github.com/georgehc/mnar_mc/blob/master/mc_algorithms.py
https://github.com/bodono/apgpy
"""

from functools import partial

import numpy as np
from sklearn.utils.extmath import randomized_svd


def std_logistic_function(x):
    return 1 / (1 + np.exp(-x))


def grad_std_logistic_function(x):
    z = np.exp(-x)
    return z / (1 + z) ** 2


def mod_logistic_function(x, gamma=3):
    x = np.clip(x, -gamma, gamma)
    one_minus_logistic_gamma = 1.0 - std_logistic_function(gamma)
    return 1 / (1 + np.exp(-x)) + 0.5 * (1 + x / gamma) * one_minus_logistic_gamma


def grad_mod_logistic_function(x, gamma=3):
    x = np.clip(x, -gamma, gamma)
    z = np.exp(-x)
    one_minus_logistic_gamma = 1.0 - std_logistic_function(gamma)
    return z / (1 + z) ** 2 + one_minus_logistic_gamma / (2 * gamma)


def one_bit_MC_fully_observed(
    M,
    link=std_logistic_function,
    link_gradient=grad_std_logistic_function,
    tau=1.0,
    gamma=3.0,
    max_rank=10,
    apg_max_iter=10,
    apg_eps=1e-12,
    apg_use_restart=True,
):
    # parameters are the same as in the paper; if `max_rank` is set to None,
    # then exact SVD is used
    m, n = M.shape
    tau_sqrt_mn = tau * np.sqrt(m * n)

    def prox(_A, t):
        _A = _A.reshape(m, n)

        # project so nuclear norm is at most tau*sqrt(m*n)
        if max_rank is None:
            U, S, VT = np.linalg.svd(_A, full_matrices=False)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
        nuclear_norm = np.sum(S)
        if nuclear_norm > tau_sqrt_mn:
            S *= tau_sqrt_mn / nuclear_norm
            _A = np.dot(U * S, VT)

        # clip matrix entries with absolute value greater than gamma
        mask = np.abs(_A) > gamma
        if mask.sum() > 0:
            _A[mask] = np.sign(_A[mask]) * gamma

        return _A.flatten()

    M_one_mask = M == 1
    M_zero_mask = M == 0

    def grad(_A):
        _A = _A.reshape(m, n)

        grad = np.zeros((m, n))
        grad[M_one_mask] = -link_gradient(_A[M_one_mask]) / link(_A[M_one_mask])
        grad[M_zero_mask] = link_gradient(_A[M_zero_mask]) / (1 - link(_A[M_zero_mask]))

        return grad.flatten()

    A_hat = solve(
        grad,
        prox,
        np.zeros(m * n),
        max_iters=apg_max_iter,
        eps=apg_eps,
        use_gra=True,
        use_restart=apg_use_restart,
        quiet=True,
    )
    P_hat = link(A_hat.reshape(m, n))
    return P_hat


def one_bit_MC_mod_fully_observed(
    M,
    link=mod_logistic_function,
    link_gradient=grad_mod_logistic_function,
    tau=1.0,
    gamma=3.0,
    max_rank=10,
    apg_max_iter=10,
    apg_eps=1e-12,
    apg_use_restart=True,
    phi=None,
):
    # parameters are the same as in the paper; if `max_rank` is set to None,
    # then exact SVD is used
    m, n = M.shape
    tau_sqrt_mn = tau * np.sqrt(m * n)
    M_zero_mask = M == 0
    if phi is None:
        phi = 0.95 * gamma

    def prox(_A, t):
        _A = _A.reshape(m, n)

        # project so nuclear norm is at most tau*sqrt(m*n)
        if max_rank is None:
            U, S, VT = np.linalg.svd(_A, full_matrices=False)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
        nuclear_norm = np.sum(S)
        if nuclear_norm > tau_sqrt_mn:
            S *= tau_sqrt_mn / nuclear_norm
            _A = np.dot(U * S, VT)

        # clip matrix entries with absolute value greater than gamma
        mask = np.abs(_A) > gamma
        if mask.sum() > 0:
            _A[mask] = np.sign(_A[mask]) * gamma

        mask = _A[M_zero_mask] > phi
        if mask.sum() > 0:
            _A[M_zero_mask][mask] = phi

        return _A.flatten()

    M_one_mask = M == 1

    def grad(_A):
        _A = _A.reshape(m, n)

        grad = np.zeros((m, n))
        grad[M_one_mask] = -link_gradient(_A[M_one_mask]) / link(
            np.maximum(_A[M_one_mask], -gamma)
        )
        grad[M_zero_mask] = link_gradient(_A[M_zero_mask]) / (
            1 - link(np.minimum(_A[M_zero_mask], phi))
        )

        return grad.flatten()

    A_hat = solve(
        grad,
        prox,
        np.zeros(m * n),
        max_iters=apg_max_iter,
        eps=apg_eps,
        use_gra=True,
        use_restart=apg_use_restart,
        quiet=True,
    )
    P_hat = link(A_hat.reshape(m, n))
    return P_hat


def npwrap(x):
    if isinstance(x, np.ndarray):
        return NumpyWrapper(x)
    return x


def npwrapfunc(f, *args):
    return npwrap(f(*args))


def solve(
    grad_f,
    prox_h,
    x_init,
    max_iters=2500,
    eps=1e-6,
    alpha=1.01,
    beta=0.5,
    use_restart=True,
    gen_plots=False,
    quiet=False,
    use_gra=False,
    step_size=False,
    fixed_step_size=False,
    debug=False,
):

    df = partial(npwrapfunc, grad_f)
    ph = partial(npwrapfunc, prox_h)

    x_init = npwrap(x_init)

    x = x_init.copy()
    y = x.copy()
    g = df(y.data)
    theta = 1.0

    if not step_size:
        # barzilai-borwein step-size initialization:
        t = 1.0 / g.norm()
        x_hat = x - t * g
        g_hat = df(x_hat.data)
        t = abs((x - x_hat).dot(g - g_hat) / (g - g_hat).norm() ** 2)
    else:
        t = step_size

    if gen_plots:
        errs = np.zeros(max_iters)

    k = 0
    err1 = np.nan
    iter_str = "iter num %i, norm(Gk)/(1+norm(xk)): %1.2e, step-size: %1.2e"
    for k in range(max_iters):

        if not quiet and k % 100 == 0:
            print(iter_str % (k, err1, t))

        x_old = x.copy()
        y_old = y.copy()

        x = y - t * g

        if prox_h:
            x = ph(x.data, t)

        err1 = (y - x).norm() / (1 + x.norm()) / t

        if gen_plots:
            errs[k] = err1

        if err1 < eps:
            break

        if not use_gra:
            theta = 2.0 / (1 + np.sqrt(1 + 4 / (theta ** 2)))
        else:
            theta = 1.0

        if not use_gra and use_restart and (y - x).dot(x - x_old) > 0:
            if debug:
                print("restart, dg = %1.2e" % (y - x).dot(x - x_old))
            x = x_old.copy()
            y = x.copy()
            theta = 1.0
        else:
            y = x + (1 - theta) * (x - x_old)

        g_old = g.copy()
        g = df(y.data)

        # tfocs-style backtracking:
        if not fixed_step_size:
            t_old = t
            t_hat = 0.5 * ((y - y_old).norm() ** 2) / abs((y - y_old).dot(g_old - g))
            t = min(alpha * t, max(beta * t, t_hat))
            if debug:
                if t_old > t:
                    print(
                        "back-track, t = %1.2e, t_old = %1.2e, t_hat = %1.2e"
                        % (t, t_old, t_hat)
                    )

    if not quiet:
        print(iter_str % (k, err1, t))
        print("terminated")
    if gen_plots:
        import matplotlib.pyplot as plt

        errs = errs[1:k]
        plt.figure()
        plt.semilogy(errs[1:k])
        plt.xlabel("iters")
        plt.title("||Gk||/(1+||xk||)")
        plt.draw()

    return x.data


class IWrapper:
    def dot(self, other):
        raise NotImplementedError("Implement in subclass")

    def __add__(self, other):
        raise NotImplementedError("Implement in subclass")

    def __sub__(self, other):
        raise NotImplementedError("Implement in subclass")

    def __mul__(self, scalar):
        raise NotImplementedError("Implement in subclass")

    def copy(self):
        raise NotImplementedError("Implement in subclass")

    def norm(self):
        raise NotImplementedError("Implement in subclass")

    @property
    def data(self):
        return self

    __rmul__ = __mul__


class NumpyWrapper(IWrapper):
    def __init__(self, nparray):
        self._nparray = nparray

    def dot(self, other):
        return np.inner(self.data, other.data)

    def __add__(self, other):
        return NumpyWrapper(self.data + other.data)

    def __sub__(self, other):
        return NumpyWrapper(self.data - other.data)

    def __mul__(self, scalar):
        return NumpyWrapper(self.data * scalar)

    def copy(self):
        return NumpyWrapper(np.copy(self.data))

    def norm(self):
        return np.linalg.norm(self.data)

    @property
    def data(self):
        return self._nparray

    __rmul__ = __mul__
