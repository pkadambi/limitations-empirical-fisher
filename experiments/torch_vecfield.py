import torch
import torch.nn as nn
import numpy as np
import eftk

theta_true = 2.
b_true = 2.


#----------------------------------------------------------------------------------------------
'''

problem setup

'''

N=1000
X, y = eftk.toydata.gradient_field_problem(N)

import matplotlib.pyplot as plt

plt.scatter(X[:,1],y)
plt.show()

#----------------------------------------------------------------------------------------------





