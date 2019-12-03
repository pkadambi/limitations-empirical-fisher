import numpy as np
from tqdm import tqdm

"""
Configuration and magic strings
"""

N_ITER = 50000
STEP_SIZE = 0.001


def run(vectorFuncs, startingPoints, quantizer=None):

    def gd(vecFunc, startingPoint, quantizer=None):
        xs = np.zeros((N_ITER, 2))
        xs[0, :] = startingPoint.copy().reshape((-1,))

        for t in tqdm(range(1, N_ITER), leave=False):
            # if t==0:
            #     import pdb
            #     pdb.set_trace()
            # print(t)
            # print(vecFunc(xs[t - 1]))
            xs[t] = (xs[t - 1] + STEP_SIZE * vecFunc(xs[t - 1])).reshape((-1,))

        if quantizer is not None:
            print('End Point:' + str(quantizer.quantize(xs[N_ITER-1])) + '\n' + '\tEnd Point FP:' + str(xs[N_ITER-1]) + '\n')

        print()

        return xs

    results = []
    for i, startingpoint in tqdm(enumerate(startingPoints), total=len(startingPoints), leave=False):
        results.append([])
        for j, vecfunc in tqdm(enumerate(vectorFuncs), total=len(vectorFuncs), leave=False):
            # import pdb
            # pdb.set_trace()
            results[i].append(gd(vecfunc, startingpoint, quantizer))
            # if j==2:
            #     exit()

    return results
