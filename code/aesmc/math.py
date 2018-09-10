import numpy as np
import scipy.misc
import torch


def logsumexp(values, dim=0, keepdim=False):
    """Logsumexp of a Tensor/Variable.

    See https://en.wikipedia.org/wiki/LogSumExp.

    input:
        values: Tensor/Variable [dim_1, ..., dim_N]
        dim: n

    output: result Tensor/Variable
        [dim_1, ..., dim_{n - 1}, 1, dim_{n + 1}, ..., dim_N] where

        result[i_1, ..., i_{n - 1}, 1, i_{n + 1}, ..., i_N] =
            log(sum_{i_n = 1}^N exp(values[i_1, ..., i_N]))
    
    
    Collapses dimension unless it's the last one?
    """

    _, idx = torch.max(values, dim=dim, keepdim=True)
    values_max = torch.gather(values, dim=dim, index=idx)
    result = torch.log(torch.sum(
        torch.exp(values - values_max.expand_as(values)),
        dim=dim,
        keepdim=keepdim
    ))

    # If keepdim==False, then we also need to get rid of the dim in values_max
    if not keepdim:
        values_max = values_max.squeeze(dim)
    result = result + values_max

    return result



def lognormexp(values, dim=0):
    """Log of exponentiates and normalizes a Tensor/Variable/np.ndarray.

    input:
        values: Tensor/Variable/np.ndarray [dim_1, ..., dim_N]
        dim: n
    output:
        result: Tensor/Variable/np.ndarray [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =

                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    if isinstance(values, np.ndarray):
        log_denominator = scipy.misc.logsumexp(
            values, axis=dim, keepdims=True
        )
        # log_numerator = values
        return values - log_denominator
    else:
        log_denominator = logsumexp(values, dim=dim, keepdim=True)
        # log_numerator = values
        return values - log_denominator.expand_as(values)
