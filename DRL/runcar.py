import sys
import gym
#import envs.arisim

from baselines import deepq
from gym.envs.registration import register

register(
    id='AirSimCarEnv-v22',
    entry_point='envs.airsim.airsimcarenv:AirSimCarEnv',
    max_episode_steps=200000,
    reward_threshold=25.0,
)



def callback(lcl, glb):
    # stop training if reward exceeds 199999
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199999
    return is_solved

def main():
    env = gym.make("AirSimCarEnv-v22")
    
    print("\n======= Act session starts for DQN Car =======")    
    trainedModel = "car.pkl"
    
    act = load_act(trainedModel)


if __name__ == '__main__':
    main()
    