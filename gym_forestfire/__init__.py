"""
Copyright 2020 Sahand Rezaei-Shoshtari. All Rights Reserved.
"""
from gym.envs.registration import register

register(
    id='ForestFire-v0',
    entry_point='gym_forestfire.envs:ForestFireEnv',
)
