import pdb

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.sisl import pursuit_v4
import supersuit as ss

from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import os
from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
from stable_baselines3.common.logger import configure

# May not all be needed
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from pantheonrl.common.wrappers import frame_wrap, recorder_wrap
from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent

from pantheonrl.algos.adap.adap_learn import ADAP
from pantheonrl.algos.adap.policies import AdapPolicyMult, AdapPolicy
from pantheonrl.algos.adap.agent import AdapAgent

from pantheonrl.algos.modular.learn import ModularAlgorithm
from pantheonrl.algos.modular.policies import ModularPolicy

from pantheonrl.algos.bc import BCShell, reconstruct_policy

from pantheonrl.envs.rpsgym.rps import RPSEnv, RPSWeightedAgent
from pantheonrl.envs.blockworldgym import simpleblockworld, blockworld
from pantheonrl.envs.liargym.liar import LiarEnv, LiarDefaultAgent

from overcookedgym.overcooked_utils import LAYOUT_LIST

import numpy as np

def generate_video(img, folder):
    directory = os.getcwd()
    print('directory', directory)
    for i in range(len(img)-50, len(img)):
        plt.imshow(img[i])
        plt.savefig(folder + "/file%02d.png" % i)

    os.chdir(folder)
    print('directory', directory)
    subprocess.call([
        'ffmpeg', '-framerate', '1', '-i', 'file%02d.png', '-r', '1', '-pix_fmt', 'yuv420p',
        'simple_collab.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

    os.chdir('../')
    print('directory', directory)
    plt.close()



def generate_env():
    env = pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=3,
        n_pursuers=2, obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
        catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)

    env = PettingZooAECWrapper(env)
    altenv = env.getDummyEnv(1)

    # framestack = 2
    # if framestack > 1:
    #     env = frame_wrap(env, framestack)
    #     altenv = frame_wrap(altenv, framestack)


    return env, altenv


### New

# env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False)
# env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')
# env = simple_push_v2.parallel_env()

# env = ss.pad_observations_v0(env)
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 4, base_class='stable_baselines3')
# env = PettingZooAECWrapper(env)
#
# print("NUMBER OF PLAYERS", env.n_players)
# # PettingZoo has many multi-player environments. To ensure that each agent
# # understands their specific observation/action space, use the getDummyEnv
# # function.
# for i in range(env.n_players - 1):
#     partner = OnPolicyAgent(PPO('MlpPolicy', env.getDummyEnv(i), verbose=1))
#     # partner = PPO('MlpPolicy', env, verbose=1)
#
#     # The second parameter ensures that the partner is assigned to a certain
#     # player number. Forgetting this parameter would mean that all of the
#     # partner agents can be picked as `player 2`, but none of them can be
#     # picked as `player 3`.
#     env.add_partner_agent(partner, player_num=i + 1)

def generate_ego():
    model_ego = PPO('MlpPolicy', env, verbose=1)
    return model_ego


def gen_partner(altenv):
    return OnPolicyAgent(PPO(policy='MlpPolicy', env=altenv, verbose=1))


def generate_partners(altenv, env, n_partners):
    partners = []
    for i in range(n_partners):
        v = gen_partner(altenv)
        print(f'Partner {i}: {v}')
        env.add_partner_agent(v)
        partners.append(v)
    return partners

env, altenv = generate_env()
print(f"Environment: {env}; Partner env: {altenv}")

ego = generate_ego()
print(f'Ego: {ego}')
partners = generate_partners(altenv, env, 1)

# ego.learn(total_timesteps=100000)
new_logger = configure('loggers/', ["stdout", "csv", "tensorboard"])
ego.set_logger(new_logger)

ego.learn(total_timesteps=20000)
ego.save('saved_models/policy')



for i in range(len(partners)):
    partners[i].model.save(f"saved_models/{i}")

