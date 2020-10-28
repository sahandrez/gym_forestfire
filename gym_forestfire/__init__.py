from gym.envs.registration import register

register(
    id='ForestFire-v0',
    entry_point='gym_forestfire.envs:ForestFireEnv',
)
