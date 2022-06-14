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

def gen_fixed(policy_type, location):
    agent = gen_load(policy_type, location)
    return StaticPolicyAgent(agent.policy)


def run_test(ego, env, num_episodes, render=False):
    rewards = []
    for game in range(num_episodes):
        obs = env.reset()
        done = False
        reward = 0
        if render:
            env.render()
        while not done:
            action = ego.get_action(obs, False)
            obs, newreward, done, _ = env.step(action)
            reward += newreward

            if render:
                env.render()
                sleep(1/60)

        rewards.append(reward)

    env.close()
    print(f"Average Reward: {sum(rewards)/num_episodes}")
    print(f"Standard Deviation: {np.std(rewards)}")





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


def gen_fixed(policy_type, location):
    agent = gen_load(policy_type, location)
    return StaticPolicyAgent(agent.policy)




env, altenv = generate_env()

print(f"Environment: {env}; Partner env: {altenv}")
# ego = generate_agent(env, args.ego, args.ego_config, args.ego_load)
# ego = gen_fixed("PPO", 'saved_models/simple_v2_ego')
ego = gen_fixed("PPO", 'saved_models/policy')
print(f'Ego: {ego}')

# alt = generate_agent(altenv, args.alt, args.alt_config, args.alt_load)
# alt =
alt = gen_fixed("PPO", f'saved_models/{0}')
env.add_partner_agent(alt)
print(f'Alt: {alt}')


run_test(ego, env, 1, False)