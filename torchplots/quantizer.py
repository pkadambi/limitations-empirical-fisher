import numpy as np
from torch.autograd.function import InplaceFunction

class UniformQuantize(InplaceFunction):
# class UniformQuantize(torch.autograd.Function):
# class UniformQuantize(nn.Module):

    @staticmethod
    # def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, eta = .01,
    #             noise=None, num_chunks=None, out_half=False):
    def forward(ctx, input, num_bits=8, min_value=None, max_value=None, eta=.01,
                    noise=None, num_chunks=None, out_half=False):

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

##
        ctx.noise = noise

        if quantize:

            ctx.num_bits = num_bits
            ctx.min_value = min_value
            ctx.max_value = max_value
            # print(input)
            # exit()
            q_weight = input.clone()

            qmin = 0.
            qmax = 2. ** num_bits - 1.
            #import pdb; pdb.set_trace()
            scale = (max_value - min_value) / (qmax - qmin)

            scale = max(scale, 1e-8)

            if FLAGS.enforce_zero:
                initial_zero_point = qmin - min_value / scale
                zero_point = 0.
                # make zero exactly represented
                #TODO: Figure out how on earth this works
                if initial_zero_point < qmin:
                    zero_point = qmin
                elif initial_zero_point > qmax:
                    zero_point = qmax
                else:
                    zero_point = initial_zero_point

                zero_point = int(zero_point)

                # output.div_(scale).add_(zero_point)
                q_weight = (q_weight / scale) + zero_point

            else:
                # output.add_(-min_value).div_(scale).add_(qmin)
                q_weight = (q_weight - min_value) / scale + qmin

            q_weight.clamp_(qmin, qmax).round_()  # quantize
            ctx.min_value = min_value
            ctx.max_value = max_value


            q_weight.add_(-qmin).mul_(scale).add_(min_value)  # dequantize

            if out_half and num_bits <= 16:
                q_weight = q_weight.half()

        else:
            # If layer is not quantized, we still need to compute
            # some type of min-max statistics on the weight kernel for noising
            q_weight=input


        # if FLAGS.regularization is not None:
        #     pert = input - q_weight
        #     ctx.save_for_backward(pert)


        return q_weight