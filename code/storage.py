import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
    # def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)

        # Computed later
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

    # def cuda(self):
    #     self.rewards = self.rewards.cuda()
    #     self.value_preds = self.value_preds.cuda()
    #     self.returns = self.returns.cuda()
    #     self.masks = self.masks.cuda()

    def to(self, device):
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)

    def insert(self, step, reward, mask):
        self.masks[step + 1].copy_(mask)
        self.rewards[step + 1].copy_(reward)

    def after_update(self):
        self.masks[0].copy_(self.masks[-1])
        self.rewards[0].copy_(self.rewards[-1])

    def compute_returns(self, next_value, gamma):
        # V_n = V(b_{t+n})
        self.returns[-1] = next_value

        # I want to ignore r_{t+0}
        # For i = n-1, n-2, ... 0
        # V_i = r_{i+1} + m_{i+1} * gamma * V_{i+1}
        # V(b_{t+n-1}) = r_{t+n} + m_{t+n} * gamma * V(b_{t+n})
        # Until V(b_{t}) [shifted by 1 compared to value_preds?!]
        for step in reversed(range(self.rewards.size(0) - 1)):
            self.returns[step] = self.returns[step + 1] * \
                gamma * self.masks[step + 1] + self.rewards[step + 1]
