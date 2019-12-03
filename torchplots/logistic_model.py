import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np


class Logistic():

    def __init__(self, X, y, n_classes, n_features, loss='logistic'):

        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.D = X.shape[1]

        self.theta = torch.tensor(np.random.randn(n_classes, n_features), requires_grad = True)
        self.b = torch.tensor(np.random.randn(n_classes))
        if loss=='logistic':
            self.criterion = nn.NLLLoss()

        #TODO: implement squared loss
        # elif loss=='squared':
        #     pass

        else:
            exit('ERROR: Choose correct loss function')

    def compute_loss(self, y_hat):
        


    def forward(self):

        pass

    def update_theta(self, lr):

        pass

    def loss(self):

        return self.criterion()

    def grad(self):

        pass

    def hess(self):

        pass

    def ef(self):
        pass

    def hess_data(self, theta):
        theta_ = theta.reshape((-1, 1))
        J_theta_f = self.X.T
        f = np.matmul(self.X, theta_).reshape((-1, 1))
        H_f_loss_diag = (h.sigmoid(f) * (1 - h.sigmoid(f))).reshape((-1, 1))
        return np.matmul(J_theta_f * H_f_loss_diag.T, J_theta_f.T) / self.N



def get_logistic_problem():
    pass

#TODO: Softmax?
class Softmax():
    def __init__(self):

        pass


class LogisticQuantized():
    def __init__(self, is_quantized=False):
        self.theta =
        self.is_quantized = is_quantized


        if quantizer is None:
            #TODO: set up the quantizer here

    def forward(self, data, is_quantized):

    def grad(self):



    def hess(self):



    def ef(self):





