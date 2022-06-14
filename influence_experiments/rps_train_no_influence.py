import pdb

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.influence_ppo import CnnPolicy, MlpPolicy, InfluencePolicy
from stable_baselines3 import PPO
from stable_baselines3 import INFLUENCE_PPO
from pettingzoo.classic import rps_v2
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

# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# import numpy as np
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

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
    env = rps_v2.env(num_actions=3, max_cycles=15)

    # env = PettingZooAECWrapper(env, ego_ind=0)
    # altenv = env.getDummyEnv(1)
    #
    # framestack = 2
    # if framestack > 1:
    #     env = frame_wrap(env, framestack)
    #     altenv = frame_wrap(altenv, framestack)
    # env = ss.concat_vec_envs_v1(env, 4, base_class='stable_baselines3')

    env = PettingZooAECWrapper(env, ego_ind=0)

    # altenv = PettingZooAECWrapper(env, ego_ind=1)
    altenv = env.getDummyEnv(1)

    # framestack = 2
    # if framestack > 1:
    #     env = frame_wrap(env, framestack)
    #     altenv = frame_wrap(env, framestack)


    return env, altenv



def generate_ego(env):
    ego_agent = PPO('MlpPolicy', env, verbose=1)
    return ego_agent


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



def gen_fixed(policy_type, location):
    agent = gen_load(policy_type, location)
    return StaticPolicyAgent(agent.policy)


def generate_agent(env, policy_type, config, location):
    if policy_type == 'DEFAULT':
        return gen_default(config, env)

    return gen_fixed(policy_type, location)

def gen_load(policy_type, location):
    if policy_type == 'PPO':
        agent = PPO.load(location)
    elif policy_type == 'ModularAlgorithm':
        agent = ModularAlgorithm.load(location)
    elif policy_type == 'BC':
        agent = BCShell(reconstruct_policy(location))
    else:
        raise EnvException("Not a valid FIXED/LOAD policy")

    return agent




def run_test(ego, env, num_episodes, render=False):
    number_to_action = {0: 'Rock', 1: 'Paper', 2: 'Scissors', 3: 'None'}


    rewards = []
    game_results = {}

    action_distribution_ego = {0:0, 1:0, 2:0, 3:0}
    action_distribution_partner = {0: 0, 1: 0, 2: 0, 3: 0}

    for game in range(num_episodes):
        game_results[game] = {'observations':[], 'rewards':[], 'actions':[], 'dones':[]}
        obs = env.reset()
        done = False
        reward = 0
        if render:
            env.render()
        while not done:
            action = ego.get_action(obs, False)
            # print("Ego action", number_to_action[action])
            output = env.step(action)
            # print("output", output)
            obs, _, newreward, done, info, all_actions = output

            # print("done", done)
            player_actions = number_to_action[all_actions[0][0]], number_to_action[all_actions[1][0]]

            action_distribution_ego[all_actions[0][0]] += 1
            action_distribution_partner[all_actions[1][0]] += 1

            # print("player_actions", player_actions)
            # print("obs", type(obs))
            new_obs = number_to_action[obs.item()]
            # print("new_obs", new_obs)
            # print()

            reward += newreward

            game_results[game]['observations'].append(new_obs)
            game_results[game]['rewards'].append(newreward)
            game_results[game]['actions'].append(player_actions)
            game_results[game]['dones'].append(done)

            if render:
                env.render()
                sleep(1/60)

        rewards.append(reward)

    env.close()

    action_distribution_ego = {action_distribution_ego[elem]/sum(action_distribution_ego.values()) for elem in action_distribution_ego}
    action_distribution_partner = {action_distribution_partner[elem] / sum(action_distribution_partner.values()) for elem in
                               action_distribution_partner}
    print(f"Average Reward: {sum(rewards)*1.0/(num_episodes*15.0)}")
    print(f"Standard Deviation: {np.std(rewards)}")
    return game_results, action_distribution_ego, action_distribution_partner

# Train Agent
def train(game_name):
    env, altenv = generate_env()
    print(f"Environment: {env}; Partner env: {altenv}")

    ego = generate_ego(env)
    print(f'Ego: {ego}')
    partners = generate_partners(altenv, env, 1)

    # ego.learn(total_timesteps=100000)
    new_logger = configure('loggers/', ["stdout", "csv", "tensorboard"])
    ego.set_logger(new_logger)

    ego.learn(total_timesteps=200000)
    ego.save(f'saved_models/{game_name}_policy')



    for i in range(len(partners)):
        partners[i].model.save(f"saved_models/{game_name}_partner_{i}")


def test(game_name, n_games):
    env, altenv = generate_env()

    print(f"Environment: {env}; Partner env: {altenv}")
    # ego = generate_agent(env, args.ego, args.ego_config, args.ego_load)
    # ego = gen_fixed("PPO", 'saved_models/simple_v2_ego')
    ego = gen_fixed("PPO", f'saved_models/{game_name}_policy')
    print(f'Ego: {ego}')

    # alt = generate_agent(altenv, args.alt, args.alt_config, args.alt_load)
    # alt =
    alt = gen_fixed("PPO", f'saved_models/{game_name}_partner_{0}')
    env.add_partner_agent(alt)
    print(f'Alt: {alt}')

    game_results, action_distribution_ego, action_distribution_partner = run_test(ego, env, n_games, False)

    # print("game_results", game_results)
    print("action_distribution_ego", action_distribution_ego)
    print("action_distribution_partner", action_distribution_partner)
    return game_results, action_distribution_ego, action_distribution_partner


if __name__ == '__main__':
    game = 'rps_influence_exp1_lstm'
    train(game)

    n_games = 10
    game_results = test(game, n_games)









