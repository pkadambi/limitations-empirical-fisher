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

    X, y = eftk.toydata.wellspecified_logreg(N)
    # pdb.set_trace()
    problem = eftk.problem_defs.LogisticRegression(X, y)
    problem.thetaStar, _ = eftk.solvers.lbfgs(problem)
    # print(problem.thetaStar)
    # exit()
    # import pdb
    # pdb.set_trace()


    #Use straight through estimator (gradients computed using quantized weight)

    q = Quantizer

    # gammas = [1 / 3, 1, 3]
    # gammas = [1 / 3, 1 / 3, 1 / 3]
    # gammas = [1 / 3, 2 / 3, 2 / 3]
    # gammas = [1 / 3, 2 / 3, 1]


    gammas = [2, 1 / 8, 4]

    def vectorFunctions(gammas, problem):
        return [
            # Hessian test (NGD)
            # lambda t: print(str(problem.loss_data(t))+'\n'+str(t)+'\n'+str(problem.hess(t)) + '\n' + str(problem.grads(t)) + '\n' + str(
            #     gammas[1] * np.linalg.solve(problem.hess(t) + (1e-8) * np.eye(2),
            #                                 problem.g(t)))+'\n'+str(problem.hess_prior())+'\n'+str(problem.hess_data(t))),

            # Hessian test (EF)
            # lambda t: print('Loss\n'+str(problem.loss_data(t))
            #                 +'\nTheta\n'+str(t)
            #                 +'\nEF\n'+str(problem.ef(t))
            #                 +'\nGradient\n' + str(np.sum(problem.grads(t), axis=0))
            #                 +'\nEF Update\n' + str(gammas[1] * np.linalg.solve(problem.hess(t) + (1e-8) * np.eye(2), problem.g(t)))),

            # GD update rule
            lambda t: - gammas[0] * problem.g(t),

            # # NGD update rule
            lambda t: - gammas[1] * np.linalg.solve(problem.hess(t) + (1e-8) * np.eye(2),
                                                    problem.g(t)),

            # EF update rule
            lambda t: - gammas[2] * np.linalg.solve(problem.ef(t) + (1e-8) * np.eye(2),
                                                    problem.g(t)),
        ]

    # gammas = [2, 1 / 12, 4]
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

    # gammas = [2, 1 / 10, 4]
    #
    # def vectorFunctions(gammas, problem):
    #     return [
    #         # GD update rule
    #         lambda t: - gammas[0] * (problem.g(q.quantize(t)) + .1 * np.reshape( 1 * (t - q.quantize(t)).T,-1)),
    #
    #         # NGD update rule
    #         lambda t: - gammas[1] * (np.linalg.solve(problem.hess(q.quantize(t)) + (1e-8) * np.eye(2),
    #                                                 problem.g(q.quantize(t))) +
    #                                  np.reshape(.25 * problem.hess(q.quantize(t)).dot(
    #                                      t - q.quantize(t)), -1)),
    #
    #         # EF update rule
    #         lambda t: - gammas[2] * (np.linalg.solve(problem.ef(q.quantize(t)) + (1e-8) * np.eye(2),
    #                                                 problem.g(q.quantize(t))) +
    #                                  np.reshape(.1 * problem.ef(q.quantize(t)).dot(
    #                                      t - q.quantize(t)), -1)),
    #     ]

    # gammas = [4, 4, 4]
    #
    # def vectorFunctions(gammas, problem):
    #     return [
    #         lambda t: - gammas[0] * np.reshape( (problem.g(q.quantize(t)) + .1 * (t - q.quantize(t)).T),-1),
    #
    #         lambda t: - gammas[1] * np.reshape( problem.g(q.quantize(t)) + .1 * problem.hess(q.quantize(t)).dot(t - q.quantize(t)), -1),
    #
    #         lambda t: - gammas[2] * np.reshape( problem.g(q.quantize(t)) + .5 * problem.ef(q.quantize(t)).dot(t - q.quantize(t)), -1),
    #     ]


    startingPoints = [
        np.array([-.5, 2]).reshape((-1, 1)),
        np.array([-1.5, -2.5]).reshape((-1, 1)),
        np.array([2, .5]).reshape((-1, 1)),
        np.array([-3, .5]).reshape((-1, 1)),
    ]


    return vectorFunctions(gammas, problem), startingPoints, X, y, problem


def run(Quantizer, savedir='./logistic_regression/vecfield.pk'):
    vectorFuncs, startingPoints, X, y, problem = load_problem(Quantizer)
    results = runner.run(vectorFuncs, startingPoints, Quantizer)
    save_results(results, savedir)


def plot(Quantizer, loaddir='./logistic_regression/vecfield.pk'):
    vectorFuncs, startingPoints, X, y, problem = load_problem(Quantizer)
    results = load_results(loaddir)
    fig = plotter.plot(X, y, problem, vectorFuncs, startingPoints, results, Quantizer)
    return [fig]


def run_appendix():
    print("This experiment has no appendix.")


def plot_appendix():
    print("This experiment has no appendix.")
