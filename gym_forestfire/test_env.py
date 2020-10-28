import gym
import numpy as np

env = gym.make('gym_forestfire:ForestFire-v0', world_size=(128, 128))
env.reset()

for _ in range(1000):
    env.render()
    s, r, d, _ = env.step(env.action_space.sample())
env.close()
