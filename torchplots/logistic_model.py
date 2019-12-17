import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import torch.nn.functional as F

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

def wellspecified_logreg(N):

    mean1 = np.array([1, 1]).reshape((-1, 1))
    mean2 = np.array([-1, -1]).reshape((-1, 1))
    cov1 = np.identity(2) * 2
    cov2 = np.identity(2) * 2

    X1 = mean1.T + np.matmul(np.random.randn(int(N / 2), 2), cov1)
    X2 = mean2.T + np.matmul(np.random.randn(int(N / 2), 2), cov2)
    y1 = np.zeros((int(N / 2), 1))
    y2 = np.ones((int(N / 2), 1))

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    return X, y

def binary_cross_entropy(yhat, y):
    # loss = -(yhat.log() * y + (1 - yhat).log() * (1 - y))
    EPS=1e-8
    yh = yhat.clone()
    yh[yh >(1-EPS)]=1-EPS
    yh[yh <EPS] = EPS

    loss = -(yh.log() * y + (1 - yh).log() * (1 - y))
    # loss = -((yhat + 1e-10).log() * y + (1 - yhat + 1e-10).log() * (1 - y))
    return loss.mean()


class LogisticModel:

    def __init__(self, N = 1000, loss='logistic'):
        np.random.seed(0)

        X, y = wellspecified_logreg(N)

        self.X = torch.tensor(X)
        self.y = torch.tensor(y).double()
        # print(self.y)
        # exit()
        self.N = N
        self.D = X.shape[1]

        self.theta = torch.tensor([5., 5.], requires_grad = True)


    def set_theta(self, theta):
        self.theta = torch.tensor(theta, requires_grad = True)
        # self.b = torch.tensor(theta[1], requires_grad = True)
        # self.theta_vec = torch.cat((self.theta, self.b), 0)

    def compute_loss(self, y_hat):
        '''

        :param y_hat:
        :return:
        '''

        return binary_cross_entropy(y_hat, self.y)

    def forward(self):

        # Data
        n_examples_per_class = 500

        # Forward Function
        # x_vec = torch.cat((self.X, torch.ones_like(x)), 1)

        z = self.X @ self.theta
        yhat = torch.sigmoid(z)
        # print(yhat)
        return yhat

        # Loss


    def compute_GD_gradient(self, theta, retain_original_theta=False):

        #Zero out gradients
        if self.theta.grad is not None:
            self.theta.grad=None

        if retain_original_theta:
            theta_orig = self.theta.clone()

        self.set_theta(theta)

        yhat = self.forward()
        lossval = self.compute_loss(y_hat=yhat)
        # print(lossval)
        lossval.backward()

        if retain_original_theta:
            self.theta = theta_orig.clone()

        return self.theta.grad.data.numpy()


    def compute_NGD_gradient(self, theta, retain_original_theta=False, gamma=1/12):

        #Zero out gradients
        if self.theta.grad is not None:
            self.theta.grad=None

        if retain_original_theta:
            theta_orig = self.theta.clone()

        self.set_theta(theta)

        yhat = self.forward()
        lossval = self.compute_loss(y_hat=yhat)

        # compute gradient (used for solving linear system for NGD update)
        lossval.backward(retain_graph=True)
        g = self.theta.grad.data.detach().numpy()

        # if self.theta.grad is not None:
        #     self.theta.grad=None

        hess = hessian(lossval, self.theta) / self.N
        # print('hess')
        hess_data = np.squeeze(hess.numpy()) * self.N
        # print('hess_')


        # In accordance with EFTK
        # hess_prior = 10.*np.eye(self.D)/self.N
        hess = hess_data

        ngd_update = .5 * 1/8 * np.linalg.solve(hess + (1e-8), g)
        # print(torch.squeeze(yhat))
        # print('Theta:')
        # print(theta)
        # print('Hessian:')
        # print(hess)
        # print('G')
        # print(g)
        # print('NGD Update:')
        # print(ngd_update)
        # print('hess data:')
        # print(hess_data)
        # print('loss')
        # print(lossval)
        # exit()

        if retain_original_theta:
            self.theta = theta_orig.clone()

        return ngd_update

    def compute_EF_gradient(self, theta, retain_original_theta=False):

        #Zero out gradients
        if self.theta.grad is not None:
            self.theta.grad=None

        if retain_original_theta:
            theta_orig = self.theta.clone()

        self.set_theta(theta)

        yhat = self.forward()
        lossval = self.compute_loss(y_hat=yhat)
        # print(lossval)
        lossval.backward()

        if retain_original_theta:
            self.theta = theta_orig.clone()

        return self.theta.grad.data.detach().numpy()

    def update_theta(self, lr):

        pass

    def loss_value(self, theta, retain_original_theta=False):
        '''

        For a specific value of theta, compute the loss.
        Note that it will update the value of theta with what's provided here

        :param theta: 2x1 torch tensor
        :return:
        '''
        if retain_original_theta:
            theta_orig = self.theta.clone()

        self.set_theta(theta)
        yhat = self.forward()
        # print(yhat)
        if retain_original_theta:
            self.theta = theta_orig.clone()

        return self.compute_loss(y_hat=yhat)

    def grad(self):

        pass

    def hess(self):

        pass

    def ef(self):
        pass

    # def hess_data(self, theta):
    #     theta_ = theta.reshape((-1, 1))
    #     J_theta_f = self.X.T
    #     f = np.matmul(self.X, theta_).reshape((-1, 1))
    #     H_f_loss_diag = (h.sigmoid(f) * (1 - h.sigmoid(f))).reshape((-1, 1))
    #     return np.matmul(J_theta_f * H_f_loss_diag.T, J_theta_f.T) / self.N



def get_logistic_problem():
    pass

#TODO: Softmax?
class Softmax():
    def __init__(self):

        pass


# class LogisticQuantized():
#     def __init__(self, is_quantized=False):
#         self.theta =
#         self.is_quantized = is_quantized
#
#
#         if quantizer is None:
#             #TODO: set up the quantizer here
#
#     def forward(self, data, is_quantized):
#
#     def grad(self):
#
#
#
#     def hess(self):
#
#
#
#     def ef(self):





