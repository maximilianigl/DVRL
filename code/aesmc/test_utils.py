import sys
import numpy as np
import torch


def num_killed_particles(A):
    '''
    input:
        A: LongTensor [batch_size, num_particles]
    output: np.array [batch_size]
    '''

    batch_size, num_particles = A.size()
    output = np.zeros(batch_size)

    for batch in range(batch_size):
        output[batch] = np.count_nonzero(
            np.bincount(A[batch].numpy(), minlength=num_particles) == 0
        )

    return output.astype(int)


# def ess(LW):
#     '''
#     Expected sample size.

#     input:
#         LW: unnormalized log weights. Tensor [batch_size, num_particles]
#     output: expected sample size. np.array [batch_size]
#     '''
#     normalized_weights_squared = math.normalize_exps(LW, dim=1)**2
#     return torch.reciprocal(torch.sum(normalized_weights_squared, dim=1).squeeze(1)).numpy()


# def num_unique_particles(A):
#     '''
#     input:
#         A: LongTensor [num_timesteps - 1, batch_size, num_particles]
#     output: np.array [batch_size]
#     '''

#     num_timesteps_minus_one, batch_size, num_particles = A.size()
#     num_timesteps = num_timesteps_minus_one + 1
#     B = LongTensor(num_timesteps, batch_size, num_particles)

#     B[num_timesteps - 1] = Range(0, num_particles - 1).squeeze(0).expand(batch_size, num_particles)

#     for t in reversed(range(num_timesteps - 1)):
#         B[t] = torch.gather(A[t], dim=1, index=B[t + 1])

#     result = np.zeros(batch_size)
#     for b in range(batch_size):
#         result[b] = np.size(np.unique(B[0][b].numpy()))

#     return result.astype(int)
