import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import grad
from numpy import inf
from logistic_model import LogisticModel
from quantizer import TorchQuantizer
import pdb

import sys


def plot_gradientDescent(ax, xs):
    # ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color=efplt.colors[optName], linewidth=4, alpha=0.9)
    ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color='k', linewidth=2, alpha=0.9)
    # ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color='k', linewidth=4, alpha=0.9)
    # ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color='k', linewidth=4, alpha=0.9)
    ax.plot(xs[0, 0], xs[0, 1], 'h', color="k", markersize=8)
    ax.plot(xs[-1, 0], xs[-1, 1], '*', color="k", markersize=12)

# GRID_DENS = 10
GRID_DENS = 20
LEVELS = 40
LABELPAD_DIFF = 5

optNames = ["GD", "NGD", "EF"]
# label_for = [
#     r"\bf{Dataset}",
#     r"\bf{GD}",
#     r"\bf{NGD}",
#     r"\bf{EF}"
# ]

# optNames = ["MSQE", "Full Hessian", "EF"]
label_for = [
    r"\bf{Dataset}",
    r"\bf{MSQE}",
    r"\bf{Full Hessian}",
    r"\bf{EF}"
]


DD = 3.5
# DD = 10
theta_lims = [-.5 - DD, -.5 + DD]
thetas = list([np.linspace(theta_lims[0], theta_lims[1], GRID_DENS) for _ in range(2)])


class Quantizer:

    def __init__(self, num_bits, q_min, q_max):

        self.qmin = q_min
        self.qmax = q_max
        self.n_bits = num_bits


        if self.n_bits<4:
            self.bins = np.linspace(self.qmin, self.qmax, 2 ** self.n_bits)
        else:
            self.bins = np.linspace(self.qmin, self.qmax, 2 ** 4)

    def quantize(self, x):

        if self.n_bits<32:
            xq = np.copy(x)

            n_intervals = 2 ** self.n_bits -1

            qrange = self.qmax - self.qmin
            scale = n_intervals / qrange


            #The following rescales x to the range between [0,1,2,..., 2^(n_bits)-1]
            xq = (xq - self.qmin) * scale

            #Quantize to [0,1,2,..., 2^(n_bits)-1]
            xq = np.clip(xq, 0, n_intervals)

            xq = np.round(xq)

            #Undo scaling
            xq = xq / scale + self.qmin

            return xq

        else:
            return x

def hide_ticks(axes):
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def hide_labels(axes):
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")


def save(fig, name):
    fig.savefig(name, bbox_inches='tight', transparent=True)


def strip_axes(axes):
    hide_ticks(axes)
    hide_labels(axes)

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


def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()


# Function defenitions
sfmx = nn.Sigmoid()
criterion = nn.BCELoss()
# criterion = nn.NLLLoss()

# Parameters
theta = torch.tensor(np.zeros(1), requires_grad=True)
b = torch.tensor(np.zeros(1), requires_grad=True)

# Data
n_examples_per_class = 500
x = torch.tensor(np.vstack([np.random.randn(n_examples_per_class ,1) + 3,
                            np.random.randn(n_examples_per_class ,1)])+2)
y = torch.cat((torch.ones(n_examples_per_class), torch.zeros(n_examples_per_class))).double()

# Forward Function
theta_vec = torch.cat((theta,b),0)
# print(theta_vec.size())
x_vec = torch.cat((x, torch.ones_like(x)),1)
# print(x_vec.size())
# exit()

z = x_vec @ theta_vec
yhat = sfmx(z)
# print(yhat)


# Loss
loss = binary_cross_entropy(yhat, y)
# loss = criterion(yhat, y)
# print(yhat)
# print(y)
# print(loss)
# exit()
# print('\nGRADIENTS')
loss.backward(retain_graph=True)
# print(theta.grad)
# print(b.grad)


# print('\nHESSIANS')
# print(jacobian(loss, theta_vec))
# print(hessian(loss, theta_vec))

for i in range(10000):
    val = i/1000
    a = torch.tensor(np.array([-val,val]), requires_grad=True)
    b = a.clone()
    ab = torch.cat((a,b),0)

    f = torch.sum(ab * ab)
    print('\nNEW ITERATION')
    print(i)
    print(jacobian(f, ab))
    print(hessian(f, ab))
    print('Done printing hessian')
    break
# exit()

def plot_loss_contour(ax, problem, losstype='CE', TEMP=4):

    def compute_losses(lossFunc):

        losses = np.zeros((len(thetas[0]), len(thetas[1])))

        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):
                theta = np.array([t1, t2]).reshape(-1, 1)
                # losses[i, j] = problem.loss_value(theta)
                losses[i, j] = lossFunc(theta)
                # print(theta)
                # print(losses[i, j])
                # print('asdf')
                # exit()
        return losses

    if losstype is 'CE':
        losses = compute_losses(lambda t: problem.ce_loss_value(t))

    elif losstype is 'DISTIL':

        losses = compute_losses(lambda t: problem.distillation_loss_value(t, T=TEMP))

    # print(thetas[0])
    # print(thetas[1])
    losses = np.array(losses)
    # print(losses)
    losses[losses==inf]=.0001
    losses[losses==.0001]=np.max(losses)

    # print(losses)
    # print(LEVELS)
    # exit()
    CS = ax.contour(thetas[0], thetas[1], losses.T, LEVELS, alpha=0.3, linewidths=2.)
    ax.clabel(CS, inline=1, fontsize=10)

def plot_vecFields(axis, problem, optimization_method = 'NGD', TEMP=4):

    def vectorField(problem, optimization_method = 'NGD', TEMP=4):
        '''
        :param problem: an instance of the problem
        :param optimization_method: Either 'GD', 'NGD', 'EF', or 'DISTIL'
        :return: The vector field to be plotted
        '''
        vector_field = [np.zeros((GRID_DENS, GRID_DENS)), np.zeros((GRID_DENS, GRID_DENS))]

        gradFunc = problem.get_grad_func(optimization_method=optimization_method)

        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):

                theta = np.array([t1, t2]).reshape(-1, 1)

                if optimization_method is 'DISTIL':
                    v = gradFunc(theta, T=TEMP)
                else:
                    v = gradFunc(theta)
                # pdb.set_trace()

                # v[v>2]=2
                # gradFunc(np.array([.5,.5]).reshape(-1,1))
                v = -np.squeeze(v)

                # print( np.squeeze(v))
                # print(np.shape(vector_field))

                for d in range(2):
                    vector_field[d][i, j] = v[d]


        vector_field = np.array(vector_field)
        # vector_field[vector_field>2.]=2.
        # print(vector_field)
        # exit()
        return vector_field

    def plot_vecField(ax, vecField, scale=30):
        U = vecField[0].copy().T
        V = vecField[1].copy().T
        cf = ax.quiver(thetas[0], thetas[1], U, V, angles='xy', scale=scale, color=[.6, .6, .6], width=0.0025, headwidth=4, headlength=3)
        return cf

    plot_loss_contour(axis, problem, TEMP=TEMP)
    vecField = vectorField(problem, optimization_method = optimization_method, TEMP=TEMP)
    plot_vecField(axis, vecField)

def fig_and_axes():
    fig = plt.figure()
    axis = plt.gca()

    # Below for single plot
    # fig = plt.figure(figsize=(12, 3.2))
    # gs = matplotlib.gridspec.GridSpec(1, 1)
    # gs.update(left=0.02, right=1, wspace=0.1, hspace=0.2, bottom=0.05, top=0.90)

    # axes = fig.add_subplot(gs[0,0])
    # return fig, axes

    return fig, axis


fig, axis = fig_and_axes()


axis.set_xlim(theta_lims)
axis.set_ylim(theta_lims)

QUANTIZE = False

fisher_method = 'NGD'

# q = Quantizer(num_bits=3, q_min=-3, q_max=2.5)
tq = TorchQuantizer(num_bits=3, q_min=-3, q_max=2.5)
if QUANTIZE:
    quantizer = tq
else:
    quantizer = None
# print(tq.bins)
# exit()
# if fisher_method in ['GD', 'NGD', '']

lr_problem = LogisticModel(quantizer = quantizer, problem='xor')
# lr_problem = LogisticModel(quantizer = quantizer)

TEMPERATURE=4


startingPoints = [
    np.array([-.5, 2]).reshape((-1, 1)),
    np.array([-1.5, -2.5]).reshape((-1, 1)),
    np.array([2, .5]).reshape((-1, 1)),
    np.array([-3, .5]).reshape((-1, 1)),
]

sequences = [np.hstack(lr_problem.train(n_iters=30000, update_method=fisher_method, starting_point=s)).T for s in startingPoints]
[plot_gradientDescent(axis, seq) for seq in sequences]
# lr_problem.train(update_method='GD', starting_point=np.array([-1.5, -1.5]))
# lr_problem.train(update_method='GD')

plot_vecFields(axis, lr_problem, optimization_method=fisher_method, TEMP=TEMPERATURE)


axis.set_xlabel('x', labelpad=LABELPAD_DIFF)
axis.set_ylabel('y', labelpad=LABELPAD_DIFF)

axis.set_xlabel(r"$\theta_1$", labelpad=LABELPAD_DIFF)
axis.set_ylabel(r"$\theta_2$", labelpad=LABELPAD_DIFF)

if QUANTIZE:
    axis.set_xticks(quantizer.bins)
    axis.set_yticks(quantizer.bins)

    axis.grid(which='major', alpha=1., linewidth=2, color='k')
else:
    axis.grid(True)

title_string = ""

if fisher_method=='GD':
    title_string = r"GD - $F(\theta)$ not used"

elif fisher_method=='NGD':
    title_string = r"NGD - $F(\theta) = \sum_n \nabla^2 log(y_n|x_n)$"
    title_string += '\n' + r"$F=\mathbb{E}[\mathrm{J}^T \frac{1}{\sigma (x)(1-\sigma(x))} \mathrm{J}]$ "

elif fisher_method=='GSQ':
    title_string = r"$Grad Squared F(\theta) = \sum_n \nabla log(y_n|x_n) \sum_n \nabla log(y_n|x_n)$"
    title_string += '\n' + r'$F=\mathbb{E}[\mathrm{J}]^T \mathbb{E}[ \mathrm{J}]$'

elif fisher_method=='EF':
    title_string = r"$EF - F(\theta) = \sum_n \nabla log(y_n|x_n) \nabla log(y_n|x_n)^T$"
    title_string = '\n' + r"$F = \mathbb{E}[\mathrm{J}^T \mathrm{J}]$"

elif fisher_method=='DISTIL':
    title_string = '\n' + r"Distillation w/GD, $F$ Not Used"

if QUANTIZE:
    title_string = 'Quantized, ' + title_string
else:
    title_string = 'FP32, ' + title_string

plt.title("2-layer Ntwk, Xor Data, \n"+title_string)


plt.show()

