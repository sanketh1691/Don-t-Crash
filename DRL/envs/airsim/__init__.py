# ref: https://github.com/openai/gym/issues/626
from gym.envs.registration import register
'''
register(
    id='AirSimCarEnv-v0',
    entry_point='envs.airsim.airsimcarenv:AirSimCarEnv',
    max_episode_steps=200000,
    reward_threshold=25.0,
)

register(
    id='AirSimDroneEnv-v0',
    entry_point='envs.airsim.airsimdroneenv:AirSimDroneEnv',
    max_episode_steps=200000,
    reward_threshold=25.0,
)
'''
