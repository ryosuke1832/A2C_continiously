from gym.envs.registration import register

register(
    id='pHRC-v0',
    entry_point='pHRC_gym.envs:pHRCEnv',
)