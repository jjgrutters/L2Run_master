# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:05:44 2021

@author: jan-g
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:14:16 2021

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

from osim.env import *
import gym

from DDPG_BC import DDPG_BC
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


# Global Callback avariable
n_steps_eval = 0
n_steps_save = 0

# Custom MLP policy of two layers of size 128 each
class CustomDDPGPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 128],
                                           layer_norm=True,
                                           feature_extraction="mlp")
     
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
    # env = DummyVecEnv([lambda: env])
    # env = DummyVecEnv([make_env_opensimrl(env_id, i) for i in range(1)])
    
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.02) * np.ones(n_actions), theta=0.1)
    model_name = "data/test"
    data_addr = "data/test"
    
    # callback to save models during training
    checkpoint_callback = CheckpointCallback(save_freq=5e4, save_path='./logs/',
                                         name_prefix='DDPG_CoT_')
    
    model = DDPG_BC(CustomDDPGPolicy, env, gamma=0.99, memory_policy=None, 
                 eval_env=None, nb_train_steps=50, nb_rollout_steps=100, 
                 nb_eval_steps=100, param_noise=None, action_noise=action_noise, 
                 normalize_observations=False, tau=0.001, batch_size=256, 
                 param_noise_adaption_interval=50, normalize_returns=False, 
                 enable_popart=False, observation_range=(-np.inf, np.inf), 
                 critic_l2_reg=0.00001,actor_l2_reg=0.00001, return_range=(-np.inf, np.inf), actor_lr=0.001, 
                 critic_lr=0.0001, clip_norm=None, reward_scale=1.0, render=False, 
                 render_eval=False, memory_limit=None, buffer_size=1000000, 
                 random_exploration=0.0, verbose=0, tensorboard_log="./l2walk_tensorboard/",
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 lambda_ac_di_loss=1.0, lambda_ac_qloss=1.0, lambda_qloss=1.0, lambda_n_step=1.0, act_prob_expert_schedule=None,
                 train_steps=0, schedule_steps=0, bc_model_name=None, dynamic_sampling_ratio=False,
                 log_addr='data/test', schedule_expert_actions=False, dynamic_loss=False, csv_log_interval=10,
                 norm_reward=1., n_expert_trajs=-1)
    
    
    model.learn(total_timesteps=1.2e6, callback=save_model_callback, seed=None, log_interval=100,
              tb_log_name="DDPG", reset_num_timesteps=True, dataset_addr= 'Expert_data_k00.npz',
              pretrain_steps=10000, max_samples_expert=None, pretrain_model_name=data_addr+"/pre_trained_model",
              replay_wrapper=None)
    
    