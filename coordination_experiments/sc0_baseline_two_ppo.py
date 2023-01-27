import copy
import pdb
import pickle
from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import os
import json
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.patches as mpatches
import torch.nn.functional as F
from torch.autograd import Variable

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.influence_ppo import CnnPolicy, MlpPolicy, InfluencePolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3 import VANILLA_PPO_SIMPLE_COLLECT_V0
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from coordination_experiments.coordination_envs import simple_collect_v0

from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper
from pantheonrl.common.wrappers import frame_wrap, recorder_wrap


DEVICE = 'cuda:0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generate_video(img, folder):
    directory = os.getcwd()
    print('directory', directory)
    for i in range(len(img) - 50, len(img)):
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


def generate_env(max_cycles):
    env = simple_collect_v0.env(max_cycles=max_cycles)

    env = PettingZooAECWrapper(env, ego_ind=0)

    altenv = env.getDummyEnv(1)

    return env, altenv


class HumanLSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(HumanLSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = 8

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("x", x.shape)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=DEVICE))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=DEVICE))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc1(h_out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.softmax(out)

        return out


def generate_ego(env, total_timesteps):
    ego_agent = OnPolicyAgent(PPO(policy='MlpPolicy', env=env, device=DEVICE, verbose=1), total_timesteps)
    return ego_agent


def gen_partner(altenv, total_timesteps):
    return OnPolicyAgent(PPO(policy='MlpPolicy', env=altenv, device=DEVICE, verbose=1), total_timesteps)


def generate_partners(altenv, env, n_partners, total_timesteps):
    partners = []
    for i in range(n_partners):
        v = gen_partner(altenv, total_timesteps)
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


def run_test(ego, env, max_cyles, num_episodes, number_to_action, render=False, render_filename='test', gif_filename='test'):

    game_results = {}
    num_episodes = 1
    # iters = 1
    for game in range(num_episodes):
        game_results[game] = {'observations': [], 'rewards': [], 'actions': [], 'dones': [], 'total_reward': 0}
        obs = env.reset()
        done = False
        reward = 0
        iter = 0
        if render:
            env.base_env.render(f'{render_filename}_{iter}')

        while not done:
            iter += 1

            action, ego_action_distr = ego.get_action(obs, False)
            # action = np.random.choice(len(ego_action_distr))

            output = env.step(action, display=render)

            obs, _, newreward, done, info, all_actions, partner_action_distr, _, _ = output

            player_actions = number_to_action[all_actions[0][0]], number_to_action[all_actions[1][0]]

            reward += newreward

            game_results[game]['observations'].append(obs)
            game_results[game]['rewards'].append(newreward)
            game_results[game]['actions'].append(player_actions)
            game_results[game]['dones'].append(done)

            if render:
                env.base_env.render(f'{render_filename}_{iter}')

        game_results[game]['total_reward'] = reward

        if render:
            env.base_env.render(f'{render_filename}_done_{iter+1}')
    env.close()

    return game_results


# Train Agent
def train(game_params):
    # device = game_params['device']
    game_name = game_params['game_name']
    max_cycles = game_params['max_cycles']
    desired_strategy = game_params['desired_strategy']
    num_iterations_per_ep = game_params['num_iterations_per_ep']
    num_interaction_episodes = game_params['num_interaction_episodes']
    num_test_games = game_params['num_test_games']
    number_to_action = game_params['number_to_action']
    action_to_one_hot = game_params['action_to_one_hot']
    with_reward_shaping = game_params['with_reward_shaping']
    default_marginal_probability = game_params['default_marginal_probability']
    partner_params = game_params['partner_params']
    transform_influence_reward = game_params['transform_influence_reward']

    env, altenv = generate_env(max_cycles)
    print(f"Environment: {env}; Partner env: {altenv}")

    ego_ppo = generate_ego(env, total_timesteps=num_iterations_per_ep)
    partners = generate_partners(altenv, env, 1, total_timesteps=num_iterations_per_ep)

    new_logger = configure('loggers/', ["stdout", "csv", "tensorboard"])
    # ego_ppo.set_logger(new_logger)

    all_game_ego_rewards = []
    all_game_partner_rewards = []
    all_game_team_rewards = []

    training_results = {}

    for ep in range(num_interaction_episodes):
        ego_ppo.learn(total_timesteps=num_iterations_per_ep)
        ego_ppo.model.save(f'{game_name}/saved_models/policy')
        print("Checkpoint 1/3: Save Ego PPO Policy")


        print("Checkpoint 2/3: Save Approximate LSTM Human Partner Models")
        for i in range(len(partners)):
            partners[i].model.save(f"{game_name}/saved_models/partner_{i}")
        print("Checkpoint 3/3: Save True PPO Partner Models")

        print("\n\n\nTesting Human Prediction LSTM..........")
        checkpoint_game_traces = test_games(game_name, action_to_one_hot, max_cycles, num_test_games, number_to_action)

        # with open(f'{game}/training_history/checkpoint_game_traces_ep{ep}.pickle', 'wb') as handle:
        #     pickle.dump(all_game_team_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
        training_results[ep] = checkpoint_game_traces

        team_rew = np.mean([checkpoint_game_traces[g]['total_reward'] for g in checkpoint_game_traces])
        all_game_team_rewards.append(team_rew)


    plt.plot(range(num_interaction_episodes), all_game_team_rewards)
    plt.legend(['team'])
    plt.xlabel("Epoch")
    plt.ylabel("Game Rewards")
    plt.title(f'{game_name}: Game Rewards')
    plt.savefig(f'{game_name}/images/game_rewards.png')
    plt.close()

    with open(f'{game}/training_history/all_game_ego_rewards.pickle', 'wb') as handle:
        pickle.dump(all_game_ego_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{game}/training_history/all_game_partner_rewards.pickle', 'wb') as handle:
        pickle.dump(all_game_partner_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{game}/training_history/all_game_team_rewards.pickle', 'wb') as handle:
        pickle.dump(all_game_team_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return training_results



def test(game_params):
    game_name = game_params['game_name']
    max_cycles = game_params['max_cycles']

    num_eval_games = game_params['num_eval_games']
    number_to_action = game_params['number_to_action']
    with_reward_shaping = game_params['with_reward_shaping']
    env, altenv = generate_env(max_cycles)

    print(f"Environment: {env}; Partner env: {altenv}")

    ego = gen_fixed("PPO", f'{game_name}/saved_models/policy')
    print(f'Ego: {ego}')

    alt = gen_fixed("PPO", f'{game_name}/saved_models/partner_{0}')
    env.add_partner_agent(alt)
    print(f'Alt: {alt}')

    render_filename = f'{game_name}/imgs_for_gif/test_gif'
    gif_filename = f'{game_name}/gifs/test_gif'

    game_results = run_test(ego, env, max_cycles, num_eval_games, number_to_action, True, render_filename, gif_filename)

    return game_results


def test_games(game_name, action_to_one_hot, max_cycles, num_episodes, number_to_action):
    env, altenv = generate_env(max_cycles)
    print(f"Environment: {env}; Partner env: {altenv}")

    ego = gen_fixed("PPO", f'{game_name}/saved_models/policy')

    alt = gen_fixed("PPO", f'{game_name}/saved_models/partner_{0}')
    env.add_partner_agent(alt)

    # Run game
    game_results = {}

    render = False
    game_history = []
    num_episodes = 1
    for game in range(num_episodes):
        game_results[game] = {'observations': [], 'rewards': [], 'actions': [], 'dones': [], 'total_reward':0}
        obs = env.reset()
        done = False

        team_reward = 0

        while not done:
            action, ego_action_distr = ego.get_action(obs, False)

            output = env.step(action, display=render)
            obs, _, newreward, done, info, all_actions, partner_action_distr, _, _ = output

            player_actions = number_to_action[all_actions[0][0]], number_to_action[all_actions[1][0]]

            team_reward += newreward

            game_results[game]['observations'].append(obs)
            game_results[game]['rewards'].append(newreward)
            game_results[game]['actions'].append(player_actions)
            game_results[game]['dones'].append(done)

            game_history.append((all_actions[0][0], all_actions[1][0]))

        game_results[game]['total_reward'] = team_reward

    if render:
        env.base_env.save_to_gif('test_baseline_two_ppo')
    env.close()
    return game_results


def transform_influence_reward(env_reward, mi_divergence, mi_guiding_divergence):
    alpha = 1
    beta = 10
    gamma = 20
    # final_reward = alpha * env_reward + beta * mi_divergence - gamma * mi_guiding_divergence
    # final_reward = alpha * env_reward - gamma * mi_guiding_divergence
    # final_reward = alpha * env_reward + beta * mi_divergence
    final_reward = env_reward
    return final_reward


if __name__ == '__main__':
    game = 'exp_results/sc0_exp1_baseline_2ppo_env-rew'
    print(f"Running Experiment: {game}")

    with_reward_shaping = True

    # Create folder for experiment data
    if not os.path.exists(game):
        # Create a new directory because it does not exist
        os.makedirs(game)

        # Create a saved models and images directory
        os.makedirs(f'{game}/saved_models')
        os.makedirs(f'{game}/images')
        os.makedirs(f'{game}/imgs_for_gif')
        os.makedirs(f'{game}/gifs')
        os.makedirs(f'{game}/images/observations')
        os.makedirs(f'{game}/training_history')
        print(f"Created folder for experiment: {game}")

    game_params = {
        'device': 'cuda:0',
        'transform_influence_reward': transform_influence_reward,
        'game_name': game,
        'max_cycles': 600,
        'with_reward_shaping': with_reward_shaping,
        # 'desired_strategy': [0.05, 0.05, 0.05, 0.05, 0.0, 0.8, 0.0],
        'desired_strategy': [0, 0, 0, 0, 0.0, 0.0, 1.0],
        'num_iterations_per_ep': 10000,
        'num_interaction_episodes': 15,
        'num_test_games': 5,
        'action_to_one_hot': {0: [1, 0, 0, 0, 0],
                              1: [0, 1, 0, 0, 0],
                              2: [0, 0, 1, 0, 0],
                              3: [0, 0, 0, 1, 0],
                              4: [0, 0, 0, 0, 1]},
        'default_marginal_probability': [0, 0, 0, 0, 0],
        'number_to_action': {0: 'N', 1: 'S', 2: 'E', 3: 'W', 4: 'NOP'},
        'number_to_color': {0: 'r', 1: 'g', 2: 'b', 3: 'm', 4: 'k'},
        'partner_params': {
            'learning_rate': 0.01,
            'input_size': 36,
            'hidden_size': 128,
            'num_layers': 1,
            'num_partner_training_epochs': 1,
            'train_percent': 0.9,
        },

        'num_eval_games': 100,
        'rewards': '''env rew'''
    }

    with open(f'{game}/experiment_parameters.pickle', 'wb') as handle:
        pickle.dump(game_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    training_results = train(game_params)

    # print("Saving Over Time Action Distributions......")
    with open(f'{game}/training_history/training_results.pickle', 'wb') as handle:
        pickle.dump(training_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    game_results = test(game_params)
    print("game_results = ", game_results)

    with open(f'{game}/training_history/testing_results.pickle', 'wb') as handle:
        pickle.dump(game_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done Running Experiment: {game}")

