import os

import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, frameskips_cases):
    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            print("Wrap Atari")
            env = make_atari(env_id)
        env.seed(seed + rank)
        if log_dir is not None:
            print("Create Monitor in {}".format(log_dir))
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            print("Wrap Deepmind")
            env = wrap_deepmind(env)
        if env_id.startswith(tuple(frameskips_cases)):
            print("Set Frameskip to 1")
            cycle_env = env
            while(True):
                if cycle_env.__class__.__name__ == 'MaxAndSkipEnv':
                    break
                cycle_env = cycle_env.env
            cycle_env._skip = 1

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            print("Wrap Pytorch")
            env = WrapPyTorch(env)
        print(env)
        return env

    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)

