import pickle as pk
import eftk
import numpy as np
import pdb

from . import runner
from . import plotter


def save_results(res):
    with open("./vecfield.pk", "wb") as fh:
        pk.dump(res, fh)


def load_results():
    with open("./vecfield.pk", "rb") as fh:
        return pk.load(fh)


def load_problem():
    np.random.seed(0)
    N = 1000

    #get the dataset
    X, y = eftk.toydata.gradient_field_problem(N)

    #get the LinearRegression object
    problem = eftk.problem_defs.LinearRegression(X, y)

    #use CG to find the one step
    problem.thetaStar, _ = eftk.solvers.cg(problem)

    #Gammas (additional multiplier to learning rate)
    gammas = [1 / 3, 1, 3]

    def vectorFunctions(gammas, problem):
        return [
            #GD update rule
            lambda t: - gammas[0] * problem.g(t),

            #NGD update rule
            lambda t: - gammas[1] * np.linalg.solve(problem.hess(t) + (1e-8) * np.eye(2), problem.g(t)),

            #EF update rule
            lambda t: - gammas[2] * np.linalg.solve(problem.ef(t) + (1e-8) * np.eye(2), problem.g(t)),
        ]

    #The four starting points in the
    startingPoints = [
        np.array([2, 4.5]).reshape((-1, 1)),
        np.array([1, 0]).reshape((-1, 1)),
        np.array([4.5, 3]).reshape((-1, 1)),
        np.array([-0.5, 3]).reshape((-1, 1)),
    ]
    print(vectorFunctions(gammas, problem))

    return vectorFunctions(gammas, problem), startingPoints, X, y, problem


def run():
    vectorFuncs, startingPoints, X, y, problem = load_problem()
    results = runner.run(vectorFuncs, startingPoints)
    save_results(results)


def plot():
    vectorFuncs, startingPoints, X, y, problem = load_problem()
    results = load_results()
    fig = plotter.plot(X, y, problem, vectorFuncs, startingPoints, results)
    return [fig]


def run_appendix():
    print("This experiment has no appendix.")


def plot_appendix():
    print("This experiment has no appendix.")
