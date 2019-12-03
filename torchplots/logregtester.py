import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import grad


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
print(loss)

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


def plot_vecFields(axes, problem, vecFuncs, with_loss_contour=True):

    def vectorField(vecFunc):
        vector_field = [np.zeros((GRID_DENS, GRID_DENS)), np.zeros((GRID_DENS, GRID_DENS))]
        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):
                theta = np.array([t1, t2]).reshape(-1, 1)
                v = vecFunc(theta)
                for d in range(2):
                    # import pdb
                    # pdb.set_trace()
                    # print(vecFunc)
                    # print(theta)
                    # print(d)
                    # print(vector_field[d][i, j])
                    # print(v[d])
                    vector_field[d][i, j] = v[d]
                    vector_field[d][i, j] = v[d]
        return vector_field

    def plot_vecField(ax, vecField, scale=30):
        U = vecField[0].copy().T
        V = vecField[1].copy().T
        cf = ax.quiver(thetas[0], thetas[1], U, V, angles='xy', scale=scale, color=efplt.grays["dark"], width=0.005, headwidth=4, headlength=3)
        return cf

    vecFields = list([vectorField(vecFuncs[i]) for i in range(len(axes))])
    for ax, vecField in zip(axes, vecFields):
        if with_loss_contour:
            plot_loss_contour(ax, problem)
        plot_vecField(ax, vecField)



