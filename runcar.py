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
    #trainedModel = "20_03_42_car_blocks_s.pkl"
    #trainedModel = "20_23_37_car_nh.pkl"
    #trainedModel = "22_13_08_car_blocks_dist.pkl"
    trainedModel = "22_19_09_car_blocks_yolo.pkl"
    
    act = deepq.load(trainedModel)
    print(act)
    while 1:
        obs, done = env.reset(), False
        print("===================================")        
        print("obs")
        print(obs)
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew) 

if __name__ == '__main__':
    main()
    