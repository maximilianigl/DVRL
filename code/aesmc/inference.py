from . import math
from . import util
from . import test_utils as tu
import time
import namedlist
import numpy as np
import torch
import logging

def sample_ancestral_index(log_weight):
    """Sample ancestral index using systematic resampling.

    input:
        log_weight: log of unnormalized weights, Tensor/Variable
            [batch_size, num_particles]
    output:
        zero-indexed ancestral index: LongTensor/Variable
            [batch_size, num_particles]
    """

    device = log_weight.device
    assert(torch.sum(log_weight != log_weight) == 0)
    batch_size, num_particles = log_weight.size()
    indices = np.zeros([batch_size, num_particles])

    uniforms = np.random.uniform(size=[batch_size, 1])
    pos = (uniforms + np.arange(0, num_particles)) / num_particles

    normalized_weights = np.exp(math.lognormexp(
        log_weight.cpu().detach().numpy(),
        dim=1
    ))
    # np.ndarray [batch_size, num_particles]
    cumulative_weights = np.cumsum(normalized_weights, axis=1)

    # trick to prevent numerical issues
    cumulative_weights = cumulative_weights / np.max(
        cumulative_weights,
        axis=1,
        keepdims=True
    )

    for batch in range(batch_size):
        indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])

    temp = torch.from_numpy(indices).long().to(device)

    return temp
