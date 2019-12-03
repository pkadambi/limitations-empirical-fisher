import pickle as pk
import eftk
import numpy as np
import pdb

from . import runner
from . import plotter

def quantize(x, n_bits, qmin=None, qmax=None):

    if qmax is None:
        qmax = max(x)

    if qmin is None:
        qmin = min(x)

    xq = np.copy(x)

    n_intervals = 2 ** n_bits -1

    qrange = qmax - qmin
    scale = n_intervals / qrange


    #The following rescales x to the range between [0,1,2,..., 2^(n_bits)-1]
    xq = (xq - qmin) * scale

    #Quantize to [0,1,2,..., 2^(n_bits)-1]
    xq = np.clip(xq, 0, n_intervals)

    xq = np.round(xq)

    #Undo scaling
    xq = xq / scale + qmin

    return xq

def save_results(res, savepath):
    # with open("vecfield.pk", "wb") as fh:
    with open(savepath, "wb") as fh:
        pk.dump(res, fh)


def load_results(loadpath):
    # with open("vecfield.pk", "rb") as fh:
    with open(loadpath, "rb") as fh:
        return pk.load(fh)


def load_problem(Quantizer):
    np.random.seed(0)
    N = 1000
    X, y = eftk.toydata.gradient_field_problem(N)
    # import pdb
    # pdb.set_trace()
    problem = eftk.problem_defs.LinearRegression(X, y)
    problem.thetaStar, _ = eftk.solvers.cg(problem)




    #Use straight through estimator (gradients computed using quantized weight)

    q = Quantizer

    # gammas = [1 / 3, 1, 3]
    # gammas = [1 / 3, 1 / 3, 1 / 3]
    # gammas = [1 / 3, 2 / 3, 2 / 3]
    # gammas = [1 / 3, 2 / 3, 1]
    # gammas = [1 / 3, 2 / 3, 3]
    #
    # def vectorFunctions(gammas, problem):
    #     return [
    #         # GD update rule
    #         lambda t: - gammas[0] * problem.g(q.quantize(t)),
    #
    #         # NGD update rule
    #         lambda t: - gammas[1] * np.linalg.solve(problem.hess(q.quantize(t)) + (1e-8) * np.eye(2),
    #                                                 problem.g(q.quantize(t))),
    #
    #         # EF update rule
    #         lambda t: - gammas[2] * np.linalg.solve(problem.ef(q.quantize(t)) + (1e-8) * np.eye(2),
    #                                                 problem.g(q.quantize(t))),
    #     ]

    gammas = [1 / 3, 1 / 3, 1 / 3]

    def vectorFunctions(gammas, problem):
        return [
            lambda t: - gammas[0] * np.reshape( (problem.g(q.quantize(t)) + .1 * (t - q.quantize(t)).T),-1),

            lambda t: - gammas[1] * np.reshape( problem.g(q.quantize(t)) + .1 * problem.hess(q.quantize(t)).dot(t - q.quantize(t)), -1),

            lambda t: - gammas[2] * np.reshape( problem.g(q.quantize(t)) + .5 * problem.ef(q.quantize(t)).dot(t - q.quantize(t)), -1),
        ]

    startingPoints = [
        np.array([2, 4.5]).reshape((-1, 1)),
        np.array([1, 0]).reshape((-1, 1)),
        np.array([4.5, 3]).reshape((-1, 1)),
        np.array([-0.5, 3]).reshape((-1, 1)),
    ]

    return vectorFunctions(gammas, problem), startingPoints, X, y, problem


def run(Quantizer, savedir='vecfield.pk'):
    vectorFuncs, startingPoints, X, y, problem = load_problem(Quantizer)
    results = runner.run(vectorFuncs, startingPoints, Quantizer)
    save_results(results, savedir)


def plot(Quantizer, loaddir='vecfield.pk'):
    vectorFuncs, startingPoints, X, y, problem = load_problem(Quantizer)
    results = load_results(loaddir)
    fig = plotter.plot(X, y, problem, vectorFuncs, startingPoints, results, Quantizer)
    return [fig]


def run_appendix():
    print("This experiment has no appendix.")


def plot_appendix():
    print("This experiment has no appendix.")
