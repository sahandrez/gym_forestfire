"""
Copyright 2020 Sahand Rezaei-Shoshtari. All Rights Reserved.
"""
import gym

env = gym.make('gym_forestfire:ForestFire-v0', world_size=(128, 128))
env.reset()

for _ in range(300):
    env.render()
    a = env.action_space.sample()
    s, r, d, _ = env.step(a)
env.close()
