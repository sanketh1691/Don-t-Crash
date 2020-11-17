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
    
    model = deepq.models.mlp([64])
    #model = deepq.models.mlp([64],layer_norm=True)
    
    print("\n======= Training session starts for DQN Car =======")    
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=10000,
        buffer_size=50000,
        exploration_fraction=1.0,   #0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        #param_noise=True,
        checkpoint_freq=2,
        learning_starts=5,
        callback=callback
    )
    trainedModel = "car.pkl"
    print("\nSaving model to", trainedModel)
    act.save(trainedModel)


if __name__ == '__main__':
    main()
    