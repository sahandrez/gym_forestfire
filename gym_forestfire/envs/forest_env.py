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
T_HORIZON = 300


class ForestFireEnv(gym.Env):

    def __init__(self, **env_kwargs):
        self.seed()
        self.reward = 0
        self.state = None
        self.t = 0

        self.forest = Forest(**env_kwargs)

        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10, shape=(STATE_H, STATE_W), dtype=np.uint8)

    def step(self, action):
        aimed_fire, is_fire = self.forest.step(action)
        self.t += 1
        step_reward = 0

        done = bool(self.t > T_HORIZON)

        # reward calculation
        # if the action has been aimed at fire: add 1 to the reward
        if aimed_fire:
            step_reward += 1
        # if fire exists but the action has done nothing: subtract 1 from the reward
        if not aimed_fire and is_fire:
            step_reward -= 1
        # if episode is over and at least 50% of the trees are remaining: add 100, otherwise: subtract 100
        if done:
            if np.mean(self.forest.world) > 0.5 * self.forest.p_init_tree:
                step_reward += 100
            else:
                step_reward -= 100

        self.reward = step_reward

        state = self.forest.world
        if state.shape != (STATE_H, STATE_W):
            state = self._scale(state, STATE_H, STATE_W)
        self.state = np.array(state)

        return self.state, step_reward, done, {}

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
