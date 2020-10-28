"""
Forest-fire gym environment.

Author: Sahand Rezaei-Shoshtari
Copyright protected.
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding

from gym_forestfire.envs.forest import Forest


STATE_W = 128
STATE_H = 128
T_HORIZON = 1000


# TODO: Compute the reward


class ForestFireEnv(gym.Env):

    def __init__(self, **env_kwargs):
        self.seed()
        self.reward = 0
        self.state = None
        self.prev_state = None
        self.t = 0

        self.forest = Forest(**env_kwargs)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def step(self, action):
        self.forest.step(action)
        self.t += 1

        # reward calculation
        self.reward = 0

        state = self.forest.world
        if state.shape != (STATE_H, STATE_W):
            state = self._scale(state, STATE_H, STATE_W)

        self.state = np.array(state)
        self.prev_state = self.state

        done = bool(not np.any(self.forest.world)
                    or self.t > T_HORIZON)

        return self.state, self.reward, done, {}

    def reset(self):
        self.forest.reset()
        self.reward = 0
        self.t = 0

        return self.step(None)[0]

    def render(self, mode='human'):
        self.forest.render()

    def close(self):
        if plt.get_fignums():
            plt.close('all')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _scale(self, im, height, width):
        original_height, original_width = im.shape
        return [[im[int(original_height * r / height)][int(original_width * c / width)]
                 for c in range(width)] for r in range(height)]
