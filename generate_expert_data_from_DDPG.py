# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:58:02 2020

@author: jan-g
"""
from osim.env import * #What is with this environment
# from DDPG.ddpg import *
# import gc
import numpy as np
from datetime import datetime
# import shap
from matplotlib import pyplot as plt

from stable_baselines import DDPG # I altered this library
from stable_baselines.common.vec_env import DummyVecEnv # and this one as well, maybe copy past them in a seperate .py 
from stable_baselines.gail import ExpertDataset, generate_expert_traj

from stable_baselines.common.vec_env import SubprocVecEnv


def generate_expert_data(model, env, n_timesteps, n_episodes, filename):
    actions_list = []
    obs_list = []
    episode_starts_list = []
    episode_returns_list = []
    rewards_list = [] 
    obs1_list = [] #obs t+1 for CoL and replay buffer
    dones_list = [] #done for CoL and replay buffer
    
    # Start by reseting the env
    obs = env.reset()
    
    for i in range(n_episodes):
        episode_starts_list.append(True)                                        # record episode starts
        for j in range(n_timesteps):
            #predict action
            action, _states = model.predict(obs)
            # save current observation
            obs_list.append(obs)
            actions_list.append(action)

            obs, reward, done, info = env.step(action)
            rewards_list.append(reward)
            obs1_list.append(obs)

            if j != 0:                                                          #If not the first, episode start = False 
                episode_starts_list.append(False)
            
            if done or j == n_timesteps - 1:                                    #If done save
                dones_list.append(True)
                obs = env.reset()                                             #If done reset env and break the forloop
                break
            
            else:
                dones_list.append(False)
                
        episode_returns_list.append(j+1)      # append length of the episodes
        print(j+1)
        
        
        # transform to np.array
        actions_arr = np.array(actions_list)
        obs_arr = np.array(obs_list)
        episode_returns_arr = np.array([episode_returns_list]).T
        episode_starts_arr = np.array([episode_starts_list]).T
        rewards_arr = np.array([rewards_list]).T     
        dones_arr = np.array([dones_list]).T
        obs1_arr = np.array(obs1_list) 
        
        # save as npz
        np.savez(filename,actions=actions_arr,
                 obs=obs_arr,rewards=rewards_arr,
                 episode_returns=episode_returns_arr,
                 episode_starts=episode_starts_arr,
                 done=dones_arr, obs1 = obs1_arr)
      

    
if __name__ == '__main__':
    env = L2RunEnv(visualize=False)
    model = DDPG.load('./BEST_POL/DDPG_CoT_10.zip')

    n_timesteps = int(500)
    n_episodes = int(20)
    filename = "Expert_data_k00"
    generate_expert_data(model, env, n_timesteps, n_episodes, filename)
    



        
