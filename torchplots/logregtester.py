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
import pdb

GRID_DENS = 15
LEVELS = 5
LABELPAD_DIFF = -20

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


DD = 3
theta_lims = [-.5 - DD, -.5 + DD]
thetas = list([np.linspace(theta_lims[0], theta_lims[1], GRID_DENS) for _ in range(2)])

# print(thetas)
# print(theta_lims)
# exit()

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
print(yhat)
print(y)
print(loss)
# exit()
print('\nGRADIENTS')
loss.backward(retain_graph=True)
print(theta.grad)
print(b.grad)


print('\nHESSIANS')
print(jacobian(loss, theta_vec))
print(hessian(loss, theta_vec))

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

def plot_loss_contour(ax, problem):

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

    losses = compute_losses(lambda t: problem.loss_value(t))
    # print(thetas[0])
    # print(thetas[1])
    losses = np.array(losses)
    # print(losses)
    losses[losses==inf]=.0001
    losses[losses==.0001]=np.max(losses)

    # print(losses)
    # print(LEVELS)
    # exit()
    ax.contour(thetas[0], thetas[1], losses.T, LEVELS, colors=["k"], alpha=0.3)

def plot_vecFields(axis, problem):

    def vectorField(problem, optimization_method = 'NGD'):
        '''
        :param problem: an instance of the problem
        :param optimization_method: Either 'GD', 'NGD', 'EF', or 'DISTIL'
        :return: The vector field to be plotted
        '''
        vector_field = [np.zeros((GRID_DENS, GRID_DENS)), np.zeros((GRID_DENS, GRID_DENS))]

        if optimization_method is 'GD':
            gradFunc = problem.compute_GD_gradient

        elif optimization_method is 'NGD':
            gradFunc = problem.compute_NGD_gradient

        elif optimization_method is 'EF':
            gradFunc = problem.compute_EF_gradient

        elif optimization_method is 'DISTIL':
            gradFunc = problem.compute_DISTIL_gradient

        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):

                theta = np.array([t1, t2]).reshape(-1, 1)
                v = gradFunc(theta)
                # pdb.set_trace()

                # v[v>2]=2

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

    plot_loss_contour(axis, problem)

    #TODO: uncomment the 2 lines below after you have figured out how to get the loss contours plotted with torch
    vecField = vectorField(problem)
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
# print(fig)
# print(axis)
# exit()

axis.set_xlim(theta_lims)
axis.set_ylim(theta_lims)

lr_problem = LogisticModel()
plot_vecFields(axis, lr_problem)


axis.set_xlabel('x', labelpad=LABELPAD_DIFF)
axis.set_ylabel('y', labelpad=LABELPAD_DIFF)

axis.set_xlabel(r"$\theta_1$", labelpad=LABELPAD_DIFF)
axis.set_ylabel(r"$\theta_2$", labelpad=LABELPAD_DIFF)

q = Quantizer(num_bits=3, q_min=-3, q_max=2.5)

axis.set_xticks(q.bins)
axis.set_yticks(q.bins)
axis.grid(which='major', alpha=1., linewidth=2, color='k')

plt.show()



startingPoints = [
    np.array([-.5, 2]).reshape((-1, 1)),
    np.array([-1.5, -2.5]).reshape((-1, 1)),
    np.array([2, .5]).reshape((-1, 1)),
    np.array([-3, .5]).reshape((-1, 1)),
]



