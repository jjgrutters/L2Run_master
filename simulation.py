# import filter_env
from osim.env import *
# from DDPG.ddpg import *
import gc
import numpy as np
# import shap
from matplotlib import pyplot as plt


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import DDPG

from stable_baselines import PPO2
from stable_baselines import PPO1
import tensorflow as tf
import time
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import FeedForwardPolicy

if __name__ == '__main__':


    env_id = "osimrl2D-v0"
    env = L2RunEnv(visualize=True)

    # Load the trained agent
    model = DDPG.load("./BEST_POL/DDPG_CoT_10.zip") #10 is best, uses knees better
    obs = env.reset()
    # rewards_list1 = []
    rewards_list2 = []


    observation = []
    for i in range(800):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards_list2.append(reward)
        observation.append(obs)
        if done:
            env.reset()
            break
            

# observations = np.asarray([observation[50], observation[20]]) # Please more than 1
# observations = observations.reshape(-1,observations.shape[1])

# def predict(inputs):
#     output = []
#     for i in range(len(inputs)):
#         output.append(model.predict(inputs[i])[0])
#     output = np.array(output)
#     return output

# explainer = shap.KernelExplainer(predict, np.zeros(observations.shape))
# shap_values = explainer.shap_values(observations)

# shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], observations[0,:], link="logit",matplotlib=True)






