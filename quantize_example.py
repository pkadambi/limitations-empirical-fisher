import numpy as np
import pdb

x = np.random.randn(20)


# q_weight = (q_weight - min_value) / scale + qmin
#
# q_weight.clamp_(qmin, qmax).round_()  # quantize
#
# q_weight.add_(-qmin).mul_(scale).add_(min_value)  # dequantize

def quantize(x, n_bits, qmin=None, qmax=None):
    if qmax is None:
        qmax = max(x)

    if qmin is None:
        qmin = min(x)

    xq = np.copy(x)

    n_intervals = 2 ** n_bits - 1

    qrange = qmax - qmin
    scale = n_intervals / qrange

    # The following rescales x to the range between [0,1,2,..., 2^(n_bits)-1]
    xq = (xq - qmin) * scale

    # Quantize to [0,1,2,..., 2^(n_bits)-1]
    xq = np.clip(xq, 0, n_intervals)

    xq = np.round(xq)

    # Undo scaling
    xq = xq / scale + qmin

    return xq


# def quantize(x, n_bits, qmin=None, qmax=None):
#
#     if qmax is None:
#         qmax = max(x)
#
#     if qmin is None:
#         qmin = min(x)
#
#     n_intervals = 2 ** n_bits -1
#
#     qrange = qmax - qmin
#
#     pdb.set_trace()
#     xq = np.copy(x)
#
#     #shift the vector such that qmin is the new zero
#     xq = xq - qmin
#
#     #The following rescales x to the range between [0,1,2,..., 2^(n_bits)-1]
#     xq = xq * n_intervals
#     xq = xq / qrange
#     xq = np.clip(xq, 0, n_intervals)
#
#     #Perform quantization to integer levels
#     xq = np.round(xq)
#
#     #Undo scaling
#     xq = xq * qrange
#     xq = xq / n_intervals
#
#     #undo shift up by qmin
#     xq = xq + qmin
#
#     return xq

print(x)
print(quantize(x, 2, -.5, .5))
print(np.clip(x, -.5, .5) - quantize(x, 2, -.5, .5))

"""
General class for problem definition

Loss(theta) = log(likelihood(theta) x prior(theta))/N
"""

import numpy as np
from . import helpers as h


class ProblemDef():

    def __init(self):
        return

    def loss(self, theta):
        return self.loss_data(theta) + self.loss_prior(theta)

    def g(self, theta):
        return np.sum(self.grads(theta), axis=0) + self.g_prior(theta)

    def hess(self, theta):
        return self.hess_data(theta) + self.hess_prior()

    def ef(self, theta):
        J = self.grads(theta) * self.N
        return np.matmul(J.T, J) / self.N

    def ggT(self, theta):
        J = self.grads(theta) * self.N
        g = J.mean(axis=0).reshape((-1, 1))
        return g @ g.T

    def covg(self, theta):
        J = self.grads(theta) * self.N
        g = self.g(theta).reshape((1, -1))
        return (J - g).T @ (J - g) / self.N

    def grads(self, theta):
        raise NotImplementedError

    def loss_data(self, theta):
        raise NotImplementedError

    def hess_data(self, theta):
        raise NotImplementedError

    def fisher(self, theta):
        raise NotImplementedError

    def loss_prior(self, theta):
        return np.linalg.norm(theta) ** 2 / (2 * self.prior_var * self.N)

    def g_prior(self, theta):
        return theta.reshape((-1,)) / (self.prior_var * self.N)

    def hess_prior(self):
        return np.eye(self.D) / (self.prior_var * self.N)


class LinearRegression(ProblemDef):
    def __init__(self, X, y, prior_var=0.1, noise=1):
        """
        X : [N x D]
        y : [N x 1]
        Fit m in y ~ N(X @ m, I), m : [D x 1]
        """
        assert (len(X.shape) == 2)

        self.X = X
        self.y = y.reshape((X.shape[0], 1))
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.prior_var = prior_var
        self.noise = noise
        self.__thetaStar = None

    def loss_data(self, theta):
        theta_ = theta.reshape((-1, 1))
        f = np.matmul(self.X, theta_)
        r = (f - self.y) ** 2 / (2 * self.noise)
        return np.mean(r)

    def grads(self, theta):
        theta_ = theta.reshape((-1, 1))
        f = np.matmul(self.X, theta_)
        r = (f - self.y)
        r = (f - np.matmul(self.X, self.thetaStar))
        J = self.X * r / (self.N * self.noise)
        return J

    def hess_data(self, theta):
        return np.matmul(self.X.T, self.X) / (self.N * self.noise)

    def fisher(self, theta):
        return self.hess_data(theta)


class LogisticRegression(ProblemDef):
    def __init__(self, X, y, prior_var=np.inf):
        """
        X : [N x D]
        y : [N x 1]
        Fit m in y ~ N(X @ m, I), m : [D x 1]
        """
        assert (len(X.shape) == 2)

        self.X = X
        self.y = y.reshape((X.shape[0], 1))
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.prior_var = prior_var

    def loss_data(self, theta):
        theta_ = theta.reshape((-1, 1))
        f = np.matmul(self.X, theta_)
        r = -np.log(h.sigmoid(f)) * self.y - np.log(h.sigmoid(-f)) * (1 - self.y)
        return r.mean()

    def grads(self, theta):
        theta_ = theta.reshape((-1, 1))
        f = np.matmul(self.X, theta_)
        J = self.X * (h.sigmoid(f) - self.y) / self.N
        return J

    def hess_data(self, theta):
        theta_ = theta.reshape((-1, 1))
        J_theta_f = self.X.T
        f = np.matmul(self.X, theta_).reshape((-1, 1))
        H_f_loss_diag = (h.sigmoid(f) * (1 - h.sigmoid(f))).reshape((-1, 1))
        return np.matmul(J_theta_f * H_f_loss_diag.T, J_theta_f.T) / self.N

    def fisher(self, theta):
        return self.hess_data(theta)
