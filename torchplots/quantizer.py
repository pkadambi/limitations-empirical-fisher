import numpy as np
from torch.autograd.function import InplaceFunction

class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=8, min_value=None, max_value=None, num_chunks=None):

        num_chunks = input.shape[0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)

        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
            # min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())

        if max_value is None:
            # max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max(-1)[0].mean(-1)  # C

        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value

        q_weight = input.clone()

        qmin = 0.
        qmax = 2. ** num_bits - 1.
        #import pdb; pdb.set_trace()
        scale = (max_value - min_value) / (qmax - qmin)

        scale = max(scale, 1e-8)

        # output.add_(-min_value).div_(scale).add_(qmin)
        q_weight = (q_weight - min_value) / scale + qmin

        q_weight.clamp_(qmin, qmax).round_()  # quantize to integers
        ctx.min_value = min_value
        ctx.max_value = max_value


        q_weight.add_(-qmin).mul_(scale).add_(min_value)  # dequantize to float quantize levels

        return q_weight


    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output

        # return grad_input
        return grad_input, None, None, None, None, None, None, None, None

#Wrapper class for quantizer function
class TorchQuantizer:
    def __init__(self, num_bits, q_min, q_max):
        self.n_bits = num_bits
        self.qmin = q_min
        self.qmax = q_max

        if self.n_bits<4:
            self.bins = np.linspace(self.qmin, self.qmax, 2 ** self.n_bits)
        else:
            self.bins = np.linspace(self.qmin, self.qmax, 2 ** 4)

    def quantize(self, x):
        return UniformQuantize().apply(x, self.n_bits, self.qmin, self.qmax)


