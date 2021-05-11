# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:41:43 2021

@author: jan-g
"""
import numpy as np
from stable_baselines.ddpg.policies import FeedForwardPolicy #THIS SHOULD NOT BE .common.policies
from stable_baselines import DDPG
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.ddpg import OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg import NormalActionNoise

from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from osim.env import *
import gym

# Custom MLP policy of two layers of size 16 each
class CustomDDPGPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 128],
                                           layer_norm=True,
                                           feature_extraction="mlp")
        
def save_replay_buffer(model, buffer_name):
    all_data = model.replay_buffer.storage.copy()
    obs_t = [all_data[i][0] for i in range(len(all_data))]
    action = [all_data[i][1] for i in range(len(all_data))]
    reward = [all_data[i][2] for i in range(len(all_data))]
    obs_tp1 = [all_data[i][3] for i in range(len(all_data))]
    done = [all_data[i][4] for i in range(len(all_data))]
    
    np.savez(buffer_name, obs_t=np.array(obs_t), action=np.array(action),
             reward=np.array(reward), obs_tp1=np.array(obs_tp1), done=np.array(done))
    
def make_env_opensimrl(env_id, rank, seed=0):

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    num_cpu = 1 # Number of processes to use
    env_id = "osimrl2D-v0"
    env = L2RunEnv(visualize=False)
    env = DummyVecEnv([lambda: env])
    env = DummyVecEnv([make_env_opensimrl(env_id, i) for i in range(1)])
    
    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(save_freq=5e4, save_path='./logs/',
                                         name_prefix='DDPG_CoT_')
   
    # Define parameter noise
    param_noise=AdaptiveParamNoiseSpec(initial_stddev=0.1, 
                                                    desired_action_stddev=0.1,
                                                    adoption_coefficient=1.01)
    
    model = DDPG(CustomDDPGPolicy, env, gamma=0.99, memory_policy=None, 
                 eval_env=None, nb_train_steps=50, nb_rollout_steps=100, 
                 nb_eval_steps=100, param_noise=param_noise, action_noise=None, 
                 normalize_observations=False, tau=0.001, batch_size=256, 
                 param_noise_adaption_interval=50, normalize_returns=False, 
                 enable_popart=False, observation_range=(-np.inf, np.inf), 
                 critic_l2_reg=0.0, return_range=(-np.inf, np.inf), actor_lr=0.0001, 
                 critic_lr=0.001, clip_norm=None, reward_scale=1.0, render=False, 
                 render_eval=False, memory_limit=None, buffer_size=1000000, 
                 random_exploration=0.0, verbose=0, tensorboard_log="./l2walk_tensorboard/",
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
    
    # Train the agent
   
    model.learn(total_timesteps=1.2e6, callback=checkpoint_callback)
    model.save('DDPG_CoT_Final')
    save_replay_buffer(model,"DDPG_replay_buffer")
