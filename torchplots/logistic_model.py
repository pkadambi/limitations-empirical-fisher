import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import scipy as sp
import torch.nn.functional as F
import pdb

def loss_fn_kd(student_soft_logprobs, teacher_soft_probs, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs from student and teacher
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # print(teacher_logits)
    # pdb.set_trace()
    # exit()
    # teacher_soft_logits = F.softmax(teacher_logits / T, dim=1)

    # teacher_soft_logits = teacher_soft_logits.float()
    # student_soft_logits = F.log_softmax(student_logits/T, dim=1)


    #For KL(p||q), p is the teacher distribution (the target distribution), and
    KD_loss = nn.KLDivLoss(reduction='mean')(student_soft_logprobs, teacher_soft_probs)
    KD_loss = (T ** 2) * KD_loss

    return KD_loss

def clip(p, threshold=10**-8):
    return np.maximum(np.minimum(p, (1.0 - threshold)), threshold)


def sigmoid(x):
    return clip(sp.special.expit(x))

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


def xor_logreg(N):

    mean_11 = np.array([1, 1]).reshape((-1, 1)) * 2.5
    mean_12 = np.array([2, 2]).reshape((-1, 1)) * 2.5

    mean_21 = np.array([2, 1]).reshape((-1, 1)) * 2.5
    mean_22 = np.array([1, 2]).reshape((-1, 1)) * 2.5

    cov = np.identity(2)

    X_11 = mean_11.T + np.matmul(np.random.randn(int(N / 4), 2), cov)
    X_12 = mean_12.T + np.matmul(np.random.randn(int(N / 4), 2), cov)

    X_21 = mean_21.T + np.matmul(np.random.randn(int(N / 4), 2), cov)
    X_22 = mean_22.T + np.matmul(np.random.randn(int(N / 4), 2), cov)

    X1 = np.vstack([X_11, X_12])
    X2 = np.vstack([X_21, X_22])

    y1 = np.zeros((int(N / 2), 1))
    y2 = np.ones((int(N / 2), 1))

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(X1[:, 0], X1[:, 1], '.r')
    # plt.plot(X2[:, 0], X2[:, 1], '.b')
    # plt.show()
    # exit()

    return X, y

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

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(X1[:, 0], X1[:, 1], '.r')
    # plt.plot(X2[:, 0], X2[:, 1], '.b')
    # plt.show()
    # exit()

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

    def __init__(self, N = 1000, problem='linear', quantizer=None):
        np.random.seed(0)

        if problem=='xor':
            X, y = xor_logreg(N)

        elif problem=='linear':
            X, y = wellspecified_logreg(N)

        self.X = torch.tensor(X)
        self.X_np = self.X.numpy()
        self.y = torch.tensor(y).double()
        self.y_np = self.y.numpy()
        # print(self.y)
        # exit()
        self.N = N
        self.D = X.shape[1]

        self.theta = torch.tensor([0., 0.], requires_grad = True)
        self.theta_star = torch.tensor([-.5,-.5], dtype=torch.double)


        self.z_star = self.X @ self.theta_star

        # Assuming temperature T=1 here
        self.teacher_soft_probs = F.sigmoid(self.z_star).view(-1, 1)
        self.teacher_soft_probs = torch.cat([self.teacher_soft_probs, 1. - self.teacher_soft_probs], dim=1)

        self.T = 1

        self.quantizer = quantizer


    def get_grad_func(self, optimization_method):

        if optimization_method is 'GD':
            gradFunc = self.compute_GD_gradient

        elif optimization_method is 'NGD':
            gradFunc = self.compute_NGD_gradient

        elif optimization_method is 'GSQ':
            gradFunc = self.compute_GSQ_gradient

        elif optimization_method is 'EF':
            gradFunc = self.compute_EF_gradient

        elif optimization_method is 'DISTIL':
            gradFunc = self.compute_distil_gradient

        return gradFunc

    def zero_gradients(self):
        # Zero out gradients
        if self.theta.grad is not None:
            self.theta.grad = None

    def set_theta(self, theta):
        self.theta = torch.tensor(theta, requires_grad = True)


        #quantization via STE is taken care of here
        if self.quantizer is not None:
            self.theta_fwd = self.quantizer.quantize(self.theta)
        else:
            self.theta_fwd = self.theta

        # self.b = torch.tensor(theta[1], requires_grad = True)
        # self.theta_vec = torch.cat((self.theta, self.b), 0)

    def compute_loss(self, y_hat):
        '''

        :param y_hat:
        :return:
        '''

        return binary_cross_entropy(y_hat, self.y)


    def forward(self):

        # Forward Function

        #Optional layer 1
        z1 = self.X @ self.theta_fwd
        z1 = torch.sigmoid(z1)

        z1 = torch.cat((z1, torch.ones_like(z1)), dim=1)
        z = z1 @ self.theta_fwd
        # z = self.X @ self.theta_fwd
        yhat = torch.sigmoid(z)
        # print(yhat)
        return yhat

        # Loss

    def train(self, n_iters=50000, update_method='GD', starting_point=None, STEP_SIZE = 0.001):

        if starting_point is None:
            self.set_theta(np.array([0., 0.]))
        else:
            self.set_theta(starting_point)

        gradFunc = self.get_grad_func(optimization_method=update_method)

        sequence = []
        print('Training Start')

        for i in range(n_iters):
            theta_t = self.theta.detach().numpy()

            theta_t = theta_t.reshape(-1,1)

            #compute gradient
            gradient = gradFunc(theta_t)
            # sequence.append(theta_next)

            #manually apply gradient
            theta_next = theta_t - STEP_SIZE * gradient


            self.set_theta(theta_next)
            sequence.append(self.theta_fwd.detach().numpy())

            if i%5000==0 or i==n_iters-1:
                print('\n\nIteration: \t'+str(i)+ '\nTheta_t: \t'+str(theta_t.reshape(-1)) + '\nGradient: \t'+str(gradient.reshape(-1)))


        sequence.append(self.theta.detach().numpy())


        print('Found Theta: ' +str(self.theta_fwd))


        return sequence


    def compute_distil_gradient(self, theta, retain_original_theta=False, T=4, alpha=1.):

        if retain_original_theta:
            theta_orig = self.theta.clone()

        self.set_theta(theta)

        #If we use a different temperature, make sure to
        if self.T != T:
            self.T = T
            self.teacher_soft_probs = F.sigmoid(self.z_star / T).view(-1, 1)
            self.teacher_soft_probs = torch.cat([self.teacher_soft_probs, 1 - self.teacher_soft_probs], dim=1)

        z = self.X @ self.theta
        yhat = torch.sigmoid(z)

        probs = F.sigmoid(z / T)
        student_soft_probs = torch.cat([probs, 1 - probs], dim=1)
        student_soft_logprobs = torch.log(student_soft_probs )

        ce_loss = self.compute_loss(y_hat=yhat)
        # pdb.set_trace()
        distill_loss = loss_fn_kd(student_soft_logprobs, self.teacher_soft_probs, T=T)
        # print(distill_loss)

        lossval = ce_loss + alpha * distill_loss

        lossval.backward()

        if retain_original_theta:
            self.theta = theta_orig.clone()

        return self.theta.grad.numpy()

    def compute_GD_gradient(self, theta, retain_original_theta=False):

        self.zero_gradients()

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


    def compute_NGD_gradient(self, theta, retain_original_theta=False, gamma=1 / 8, diag_load=1e-5):

        self.zero_gradients()

        if retain_original_theta:
            theta_orig = self.theta.clone()

        #Set new theta
        self.set_theta(theta)

        yhat = self.forward()
        lossval = self.compute_loss(y_hat=yhat)

        # compute gradient (used for solving linear system for NGD update)
        lossval.backward(retain_graph=True)
        g = self.theta.grad.data.detach().numpy()


        # compute Hessian
        hess = hessian(lossval, self.theta)
        hess_data = np.squeeze(hess.numpy())
        hess = hess_data

        # Solve linear system (invert hessian)
        ngd_update = gamma * np.linalg.solve(hess + np.eye(2) * diag_load, g)


        if retain_original_theta:
            self.theta = theta_orig.clone()

        # print('Gradient')
        # print(g)
        # print('Hess')
        # print(hess)
        # print('EF Update')
        # print(ngd_update)

        return ngd_update


    def compute_GSQ_gradient(self, theta, retain_original_theta=False, diag_load=1e-5):

        self.zero_gradients()

        if retain_original_theta:
            theta_orig = self.theta.clone()

        #Set new theta
        self.set_theta(theta)
        yhat = self.forward()
        lossval = self.compute_loss(y_hat=yhat)

        # compute gradient (used for solving linear system for NGD update)
        lossval.backward(retain_graph=True)
        g = self.theta.grad.data.detach().numpy()

        # compute Hessian
        hess = g @ g.T

        # Solve linear system (invert hessian)
        # ngd_update = .005 * 1/8 * np.linalg.solve(hess + np.eye(2) * (1e-8), g)
        ngd_update = 1/4 * np.linalg.solve(hess + np.eye(2) * diag_load, g)

        # print('Gradient')
        # print(g)
        # print(np.shape(g))
        #
        # print('Hess')
        # print(np.shape(hess))
        # print(hess)
        #
        # print('EF Update')
        # print(ngd_update)
        # exit()

        if retain_original_theta:
            self.theta = theta_orig.clone()

        return ngd_update

    def compute_EF_gradient(self, theta,  retain_original_theta=False, gamma=4):

        if retain_original_theta:
            theta_orig = self.theta.clone()


        self.set_theta(theta)


        # self.zero_gradients()
        g = self.grads(theta)
        g = np.sum(g, axis=0)
        hess = self.ef(theta)


        ngd_update = gamma * np.linalg.solve(hess + np.eye(2) * (1e-8), g)
        # ngd_update

        if retain_original_theta:
            self.theta = theta_orig.clone()

        return ngd_update.reshape(-1,1)


    def ef(self, theta):
        J = self.grads(theta) * self.N
        return np.matmul(J.T, J) / self.N

    def grads(self, theta):

        theta_ = theta.reshape((-1, 1))
        f = np.matmul(self.X_np, theta_)
        J = self.X_np * (sigmoid(f) - self.y_np) / self.N
        return J

    def update_theta(self, lr):

        pass

    def distillation_loss_value(self, theta, retain_original_theta=False, T=1):

        if retain_original_theta:
            theta_orig = self.theta.clone()
        self.set_theta(theta)

        if self.T != T:
            self.T = T
            self.teacher_soft_probs = F.sigmoid(self.z_star / self.T)
            self.teacher_soft_probs = self.teacher_soft_probs.view(-1, 1)

        z = self.X @ self.theta
        student_soft_logprobs = F.logsigmoid(z / T)


        distill_loss = loss_fn_kd(student_soft_logprobs, self.teacher_soft_probs, T=T)

        if retain_original_theta:
            self.theta = theta_orig.clone()

        return distill_loss

    def ce_loss_value(self, theta, retain_original_theta=False):
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





