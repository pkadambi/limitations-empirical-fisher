import numpy as np
from efplt import plt
from efplt import matplotlib
import efplt
import pdb

GRID_DENS = 15
LEVELS = 5
LABELPAD_DIFF = 0

optNames = ["GD", "NGD", "EF"]
label_for = [
    r"\bf{Dataset}",
    r"\bf{GD}",
    r"\bf{NGD}",
    r"\bf{EF}"
]

# optNames = ["MSQE", "Full Hessian", "EF"]
# label_for = [
#     r"\bf{Dataset}",
#     r"\bf{MSQE}",
#     r"\bf{Full Hessian}",
#     r"\bf{EF}"
# ]
label_for = [
    r"\bf{Distillation}",
    r"\bf{MSQE}",
    r"\bf{Full Hessian}",
    r"\bf{EF}"
]

DD = 3
theta_lims = [-.5 - DD, -.5 + DD]
thetas = list([np.linspace(theta_lims[0], theta_lims[1], GRID_DENS) for _ in range(2)])


def fig_and_axes():
    fig = plt.figure(figsize=(12, 3.2))
    gs = matplotlib.gridspec.GridSpec(1, 4)
    gs.update(left=0.02, right=1, wspace=0.1, hspace=0.2, bottom=0.05, top=0.90)
    axes = [fig.add_subplot(gs[i, j]) for i in range(1) for j in range(4)]
    return fig, axes


def plot_dataset(ax, problem, X, y):
    # pdb.set_trace()

    ax.plot(X[:, 1], X[:, 1] * problem.thetaStar[1] + X[:, 0] * problem.thetaStar[0], '.',
            label=r"$y = \theta_1 x_1 + \theta_2 x_2$", markersize=3, alpha=0.4, color=efplt.colors_classes[0])



    # ax.set_xlim([0, 10])
    # ax.set_ylim([0, 20])

    # xs = np.linspace(0, 9, 100)
    # ax.plot(xs, xs * problem.thetaStar[1] + problem.thetaStar[0], '--', label=r"$y = \theta x + b$", linewidth=3, color="k")

    # ax.set_xlim([-1, 10])
    # ax.set_ylim([-2, 22])

    ax.legend(prop={'size': 16}, borderpad=0.3)


def plot_loss_contour(ax, problem):
    def compute_losses(lossFunc):
        losses = np.zeros((len(thetas[0]), len(thetas[1])))
        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):
                theta = np.array([t1, t2]).reshape(-1, 1)
                losses[i, j] = lossFunc(theta)
        return losses

    losses = compute_losses(lambda t: problem.loss(t))
    ax.contour(thetas[0], thetas[1], losses.T, LEVELS, colors=["k"], alpha=0.3)


def plot_vecFields(axes, problem, vecFuncs, with_loss_contour=True):

    def vectorField(vecFunc):
        vector_field = [np.zeros((GRID_DENS, GRID_DENS)), np.zeros((GRID_DENS, GRID_DENS))]
        for i, t1 in enumerate(thetas[0]):
            for j, t2 in enumerate(thetas[1]):
                theta = np.array([t1, t2]).reshape(-1, 1)
                v = vecFunc(theta)
                for d in range(2):
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


def plot_gradientDescent(ax, xs, optName):
    # ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color=efplt.colors[optName], linewidth=4, alpha=0.9)
    ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color='k', linewidth=4, alpha=0.9)
    # ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color='k', linewidth=4, alpha=0.9)
    # ax.plot(xs[:, 0], xs[:, 1], linestyle='-', color='k', linewidth=4, alpha=0.9)
    ax.plot(xs[0, 0], xs[0, 1], 'h', color="k", markersize=8)
    ax.plot(xs[-1, 0], xs[-1, 1], '*', color="k", markersize=12)

def plot_quantization_grid(ax):

    return 0


def plot(X, y, problem, vecFuncs, startingPoints, results, Quantizer):

    fig, axes = fig_and_axes()

    plot_vecFields(axes[1:], problem, vecFuncs)
    plot_dataset(axes[0], problem, X, y)

    for i, startingpoint in enumerate(startingPoints):
        for j, vecfunc in enumerate(vecFuncs):
            plot_gradientDescent(axes[1 + j], results[i][j], optNames[j])

    axes[0].set_xlabel(r"$x_1$", labelpad=LABELPAD_DIFF)
    axes[0].set_ylabel(r"$y (Pre-sigmoid)$", labelpad=LABELPAD_DIFF)

    q = Quantizer

    for ax in axes[1:]:
        ax.set_xlim(theta_lims)
        ax.set_ylim(theta_lims)
        # ax.set_xticks(theta_lims)
        # ax.set_yticks(theta_lims)

        #plot quantizer grid

        ax.set_xlabel(r"$\theta_0$", labelpad=LABELPAD_DIFF)
        ax.set_ylabel(r"$\theta_1$", labelpad=LABELPAD_DIFF)



        ax.set_xticks(q.bins)
        ax.set_yticks(q.bins)

        empty_string_labels = [''] * len(q.bins)
        ax.set_xticklabels(empty_string_labels)
        ax.set_yticklabels(empty_string_labels)

        # y_axis = ax.get_yaxis()



        # efplt.strip_axes(axes)

        ax.grid(which='major', alpha=1., linewidth=2, color='k')

    for i, ax in enumerate(axes):
        ax.set_title(label_for[i])

    return fig
