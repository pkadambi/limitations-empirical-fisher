import numpy as np

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