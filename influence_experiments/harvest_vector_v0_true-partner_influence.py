import copy
import pdb
import pickle
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.influence_ppo import CnnPolicy, MlpPolicy, InfluencePolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy

from stable_baselines3 import PPO
from stable_baselines3 import INFLUENCE_PPO_HARVEST_VECTOR_V1_TRUE_PARTNER
# from pettingzoo.classic import rps_v2
# from influence_envs.joint_particle import joint_particle_env
from influence_experiments.influence_envs import harvest_vector_v0
# import supersuit as ss

from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import os
import json
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
import matplotlib.patches as mpatches
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

DEVICE = 'cuda:0'
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    env = harvest_vector_v0.env(max_cycles=max_cycles)

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


class EgoRPSAgent():
    def __init__(self, env, action_to_one_hot, default_marginal_probability, desired_strategy, partner_params):
        self.ego_model = INFLUENCE_PPO_HARVEST_VECTOR_V1_TRUE_PARTNER('MlpPolicy', env, device=DEVICE, verbose=1)
        # self.ego_model = PPO(policy='MlpPolicy', env=env, device=DEVICE, verbose=1)

        # print("initial action_to_one_hot", action_to_one_hot)
        self.ego_model.set_reward_params(action_to_one_hot, default_marginal_probability, desired_strategy)

        # Set episode index for saving old data
        self.past_game_data = {}
        self.eps_index = 0

        # Save environment
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # pdb.set_trace()
        # Construct partner model

        self.partner_params = partner_params
        self.learning_rate = partner_params['learning_rate']
        self.input_size = partner_params['input_size']
        self.hidden_size = partner_params['hidden_size']
        self.num_layers = partner_params['num_layers']
        self.train_percent = partner_params['train_percent']

        self.num_classes = self.action_space.n  # 4
        self.num_partner_training_epochs = partner_params['num_partner_training_epochs']

        # self.seq_length = 10
        self.initialize_partner_model()
        # self.batch_size = 10

        self.action_to_one_hot = action_to_one_hot

    def add_game_to_train(self, eps_index, game_result):
        self.past_game_data[eps_index] = game_result
        self.eps_index = eps_index

    def initialize_partner_model(self):
        self.partner_model = HumanLSTM(self.num_classes, self.input_size, self.hidden_size, self.num_layers)
        self.partner_model.to(DEVICE)
        self.partner_model.train()
        # self.partner_model.double()
        self.criterion = nn.CrossEntropyLoss()
        # use LBFGS as optimizer since we can load the whole data to train
        # self.optimizer = optim.LBFGS(self.partner_model.parameters(), lr=0.8)
        self.optimizer = optim.Adam(self.partner_model.parameters(), lr=self.learning_rate)

    def train_partner_model(self):
        train_x, train_y = self.construct_dataset()
        train_x, train_y = np.expand_dims(np.array(train_x), axis=1), np.array(train_y)
        # train_x, train_y = np.array(train_x), np.array(train_y)

        dataset_size = train_x.shape[0]
        train_size = int(dataset_size * self.train_percent)
        test_size = dataset_size - train_size

        dataX = Variable(torch.Tensor(train_x))
        dataY = Variable(torch.Tensor(train_y))

        trainX = Variable(torch.Tensor(np.array(train_x[0:train_size]))).to(device=DEVICE)
        trainY = Variable(torch.Tensor(np.array(train_y[0:train_size]))).to(device=DEVICE)

        testX = Variable(torch.Tensor(np.array(train_x[train_size:dataset_size])))
        testY = Variable(torch.Tensor(np.array(train_y[train_size:dataset_size])))

        # np.swapaxes(trainX, 1, 2)
        print("trainX", trainX.shape)
        print("trainY", trainY.shape)

        # Train the model
        for epoch in range(self.num_partner_training_epochs):
            # trainX.double()
            # trainY.double()
            # pdb.set_trace()
            outputs = self.partner_model(trainX)
            self.optimizer.zero_grad()

            # obtain the loss function
            # print("outputs", outputs)
            # print("trainY", trainY)
            trainY = trainY.type(torch.LongTensor)
            loss = self.criterion(outputs, trainY.to(device=DEVICE))

            loss.backward()

            self.optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    def construct_dataset(self):
        # TODO: This needs to be more general
        # game_outcomes.append((state_t, record_ego_action, record_partner_action, state_t1, record_reward))

        train_x, train_y = [], []
        game_data = self.past_game_data[self.eps_index]
        for i in range(1, len(game_data) - 1):
            state_i, ego_action_i, partner_action_i, state_i1, record_reward = game_data[i]

            _, _, partner_action_i1, state_i2, _ = game_data[i + 1]

            # add_x = np.concatenate([ego_action_i, state_i, state_i1], axis=0)
            # pdb.set_trace()
            # ego_action_i = copy.deepcopy(ego_action_i)
            # pdb.set_trace()
            # ego_action_i.extend([[0, 0, 0]])
            # ego_action_i = np.reshape(ego_action_i, (1, 10))
            # ego_action_i = np.broadcast_to(ego_action_i, (1, 10, 10))
            # pdb.set_trace()
            # state_i = np.squeeze(state_i, axis=2)
            # state_i = np.squeeze(state_i, axis=0)
            #
            # state_i1 = np.squeeze(state_i1, axis=2)
            # state_i1 = np.squeeze(state_i1, axis=0)

            add_x = np.concatenate([ego_action_i, state_i, state_i1], axis=0)  # length = (14,)
            # add_x = np.expand_dims(np.array([add_x]), axis=0)
            # print("add_x",add_x.shape)
            add_y = partner_action_i1.index(1)  # Not one hot encoded version

            train_x.append(add_x)
            train_y.append(add_y)

        train_x, train_y = np.array(train_x), np.array(train_y)

        print("train_x", train_x.shape)
        print("train_y", train_y.shape)
        return train_x, train_y


def generate_ego(env, action_to_one_hot, default_marginal_probability, desired_strategy, partner_params):
    ego_agent = EgoRPSAgent(env, action_to_one_hot, default_marginal_probability, desired_strategy, partner_params)
    # ego_agent = OnPolicyAgent(PPO(policy='MlpPolicy', env=env, device=DEVICE, verbose=1))
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
    if policy_type == 'INFLUENCE_PPO':
        agent = INFLUENCE_PPO_HARVEST_VECTOR_V1_TRUE_PARTNER.load(location)
    elif policy_type == 'PPO':
        agent = PPO.load(location)
    elif policy_type == 'ModularAlgorithm':
        agent = ModularAlgorithm.load(location)
    elif policy_type == 'BC':
        agent = BCShell(reconstruct_policy(location))
    else:
        raise EnvException("Not a valid FIXED/LOAD policy")

    return agent


def run_test(ego, env, max_cyles, num_episodes, number_to_action, render=False):
    # number_to_action = {0: 'Cooperate', 1: 'Defect', 2: 'None'}

    rewards = []
    game_results = {}

    num_actions = env.action_space.n

    action_distribution_ego = {a: 0 for a in range(num_actions)}
    action_distribution_partner = {a: 0 for a in range(num_actions)}

    partner_obs = None
    # render = True

    # num_episodes = 1
    # iters = 1
    for game in range(num_episodes):
        game_results[game] = {'observations': [], 'rewards': [], 'actions': [], 'dones': []}
        obs = env.reset()
        done = False
        reward = 0
        if render:
            env.base_env.render()
        while not done:
            # if iters > 10:
            #     break
            # iters += 1
            action, ego_action_distr = ego.get_action(obs, False)
            # print("Ego action", number_to_action[action])
            initial_obs = obs
            output = env.step(action)
            # print("env")
            # print("output", output)
            obs, _, newreward, done, info, all_actions, partner_action_distr, _ = output
            # print(f"initial obs: {initial_obs}, actions: {all_actions} ")
            # print("done", done)
            player_actions = number_to_action[all_actions[0][0]], number_to_action[all_actions[1][0]]
            # print(f"player_actions: {player_actions}: ",(all_actions[0][0], all_actions[1][0]))

            action_distribution_ego[all_actions[0][0]] += 1
            action_distribution_partner[all_actions[1][0]] += 1

            # print("player_actions", player_actions)
            # print("obs", type(obs))
            # new_obs = number_to_action[obs.item()]
            # partner_obs = action
            # print("new_obs", new_obs)
            # print()

            reward += newreward

            game_results[game]['observations'].append(obs)
            game_results[game]['rewards'].append(newreward)
            game_results[game]['actions'].append(player_actions)
            game_results[game]['dones'].append(done)

            if render:
                env.base_env.render()
                # sleep(1/60)

        rewards.append(reward)

    env.close()

    # action_distribution_ego = {action_distribution_ego[elem]/sum(action_distribution_ego.values()) for elem in action_distribution_ego}
    # action_distribution_partner = {action_distribution_partner[elem] / sum(action_distribution_partner.values()) for elem in
    #                            action_distribution_partner}

    # print("action_distribution_ego", action_distribution_ego)
    # print("action_distribution_partner", action_distribution_partner)
    # pdb.set_trace()

    ego_action_sum = sum(action_distribution_ego.values())
    partner_action_sum = sum(action_distribution_partner.values())
    normalized_action_distribution_ego = {k: action_distribution_ego[k] / ego_action_sum for k in
                                          action_distribution_ego}

    normalized_partner_distribution_ego = {k: action_distribution_partner[k] / partner_action_sum for k in
                                           action_distribution_partner}

    print(f"Average Reward: {sum(rewards) * 1.0 / (num_episodes * max_cyles)}")
    print(f"Standard Deviation: {np.std(rewards)}")
    return game_results, normalized_action_distribution_ego, normalized_partner_distribution_ego


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
    default_marginal_probability = game_params['default_marginal_probability']
    partner_params = game_params['partner_params']
    transform_influence_reward = game_params['transform_influence_reward']

    env, altenv = generate_env(max_cycles)
    print(f"Environment: {env}; Partner env: {altenv}")

    ego_agent = generate_ego(env, action_to_one_hot, default_marginal_probability, desired_strategy, partner_params)
    ego_ppo = ego_agent.ego_model
    partners = generate_partners(altenv, env, 1, total_timesteps=num_iterations_per_ep)
    ego_ppo.set_true_partner(partners[0].model)

    new_logger = configure('loggers/', ["stdout", "csv", "tensorboard"])
    ego_ppo.set_logger(new_logger)

    partner_prediction_accuracies = []

    over_time_partner_action_distribution = {}
    over_time_ego_action_distribution = {}

    all_game_ego_rewards = []
    all_game_partner_rewards = []
    all_game_team_rewards = []

    for ep in range(num_interaction_episodes):
        ego_ppo.set_partner_model(ego_agent.partner_model, transform_influence_reward, DEVICE)

        ego_ppo.set_episode_instance(ep)
        ego_ppo.learn(total_timesteps=num_iterations_per_ep)
        ego_ppo.save(f'{game_name}/saved_models/policy')
        print("Checkpoint 1/3: Save Ego PPO Policy")

        # print('all_game_data_by_ep', ego_ppo.all_game_data_by_ep)
        # episode_data = ego_ppo.all_game_data_by_ep[ep]
        # ego_agent.add_game_to_train(ep, episode_data)
        # pdb.set_trace()
        # ego_agent.train_partner_model()

        # torch.save(ego_agent.partner_model.state_dict(), f'{game_name}/saved_models/partner_model')
        print("Checkpoint 2/3: Update EGO with TRUE Partner Model")
        ego_ppo.set_true_partner(partners[0].model)

        for i in range(len(partners)):
            partners[i].model.save(f"{game_name}/saved_models/partner_{i}")
        print("Checkpoint 3/3: Save True PPO Partner Models")

        print("\n\n\nTesting Human Prediction LSTM..........")
        partner_prediction_accuracy, partner_action_distribution, ego_action_distribution, game_ego_rewards, game_partner_rewards, game_team_rewards = \
            test_games(game_name, partner_params, action_to_one_hot, max_cycles, num_test_games, number_to_action)

        # print(f'episode {ep}: partner_prediction_accuracy = {partner_prediction_accuracy}')
        print("partner_action_distr", partner_action_distribution)
        print("ego_action_distribution", ego_action_distribution)

        # partner_prediction_accuracies.append(partner_prediction_accuracy)
        over_time_partner_action_distribution[ep] = partner_action_distribution
        over_time_ego_action_distribution[ep] = ego_action_distribution
        all_game_ego_rewards.append(np.mean(game_ego_rewards))
        all_game_partner_rewards.append(np.mean(game_partner_rewards))
        all_game_team_rewards.append(np.mean(game_team_rewards))

    # plt.plot(range(num_interaction_episodes), partner_prediction_accuracies)
    # plt.xlabel("Epoch")
    # plt.ylabel("Partner Prediction Accuracy")
    # plt.title(f'{game_name}: Partner Prediction Accuracy')
    # plt.savefig(f'{game_name}/images/partner_pred_acc.png')
    # plt.close()

    plt.plot(range(num_interaction_episodes), all_game_ego_rewards)
    plt.plot(range(num_interaction_episodes), all_game_partner_rewards)
    plt.plot(range(num_interaction_episodes), all_game_team_rewards)
    plt.legend(['ego', 'partner', 'team'])
    plt.xlabel("Epoch")
    plt.ylabel("Game Rewards")
    plt.title(f'{game_name}: Game Rewards')
    plt.savefig(f'{game_name}/images/game_rewards.png')
    plt.close()

    plt.plot(range(num_interaction_episodes), all_game_ego_rewards)
    plt.plot(range(num_interaction_episodes), all_game_partner_rewards)
    plt.plot(range(num_interaction_episodes), all_game_team_rewards)
    plt.legend(['ego', 'partner', 'team'])
    plt.xlabel("Epoch")
    plt.ylabel("Game Rewards")
    max_y = max([max(all_game_team_rewards), max(all_game_ego_rewards), max(all_game_partner_rewards)])
    plt.ylim(-1, max_y + 2)
    plt.title(f'{game_name}: Game Rewards')
    plt.savefig(f'{game_name}/images/game_rewards_0tomax.png')
    plt.close()

    with open(f'{game}/training_history/all_game_ego_rewards.pickle', 'wb') as handle:
        pickle.dump(all_game_ego_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{game}/training_history/all_game_partner_rewards.pickle', 'wb') as handle:
        pickle.dump(all_game_partner_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{game}/training_history/all_game_team_rewards.pickle', 'wb') as handle:
        pickle.dump(all_game_team_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return over_time_partner_action_distribution, over_time_ego_action_distribution


def plot_action_distribution(game_name, action_distribution_ego, action_distribution_partner, number_to_action):
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    ego_dist = list(action_distribution_ego.values())
    part_dist = list(action_distribution_partner.values())

    # Set position of bar on X axis
    br1 = np.arange(len(number_to_action))
    br2 = [x + barWidth for x in br1]
    # br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, ego_dist, color='r', width=barWidth,
            edgecolor='grey', label='Ego')
    plt.bar(br2, part_dist, color='g', width=barWidth,
            edgecolor='grey', label='Partner')

    # Adding Xticks
    plt.xlabel('RPD Action', fontweight='bold', fontsize=15)
    plt.ylabel('Percent of Actions in 150 games', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(number_to_action))], list(number_to_action.values()))

    plt.title(f"Trial {game_name}: Action Distributions")
    plt.legend()
    plt.savefig(f'{game_name}/images/TEST_action_distrib.png')
    plt.close()


def test(game_params):
    game_name = game_params['game_name']
    max_cycles = game_params['max_cycles']
    desired_strategy = game_params['desired_strategy']
    num_iterations_per_ep = game_params['num_iterations_per_ep']
    num_interaction_episodes = game_params['num_interaction_episodes']
    num_test_games = game_params['num_test_games']
    action_to_one_hot = game_params['action_to_one_hot']
    default_marginal_probability = game_params['default_marginal_probability']
    partner_params = game_params['partner_params']
    num_eval_games = game_params['num_eval_games']
    number_to_action = game_params['number_to_action']
    env, altenv = generate_env(max_cycles)

    print(f"Environment: {env}; Partner env: {altenv}")

    # ego = generate_agent(env, args.ego, args.ego_config, args.ego_load)
    # ego = gen_fixed("PPO", 'saved_models/simple_v2_ego')
    ego = gen_fixed("INFLUENCE_PPO", f'{game_name}/saved_models/policy')
    print(f'Ego: {ego}')

    # alt = generate_agent(altenv, args.alt, args.alt_config, args.alt_load)
    # alt =
    alt = gen_fixed("PPO", f'{game_name}/saved_models/partner_{0}')
    env.add_partner_agent(alt)
    print(f'Alt: {alt}')

    game_results, action_distribution_ego, action_distribution_partner = run_test(ego, env, max_cycles, num_eval_games,
                                                                                  number_to_action, False)

    # print("game_results", game_results)
    print("action_distribution_ego", action_distribution_ego)
    print("action_distribution_partner", action_distribution_partner)

    plot_action_distribution(game_name, action_distribution_ego, action_distribution_partner, number_to_action)

    return game_results, action_distribution_ego, action_distribution_partner


def test_games(game_name, partner_params, action_to_one_hot, max_cycles, num_episodes, number_to_action):
    env, altenv = generate_env(max_cycles)
    print(f"Environment: {env}; Partner env: {altenv}")

    ego = gen_fixed("PPO", f'{game_name}/saved_models/policy')

    alt = gen_fixed("PPO", f'{game_name}/saved_models/partner_{0}')
    env.add_partner_agent(alt)

    # Run game

    ego_rewards = []
    partner_rewards = []
    team_rewards = []

    game_results = {}

    partner_obs = None
    partner_action_distr = None
    ego_action_distr = None

    partner_action_distribution = {}
    ego_action_distribution = {}

    render = False
    state_t_minus_1 = None
    game_history = []
    for game in range(num_episodes):
        game_results[game] = {'observations': [], 'rewards': [], 'actions': [], 'dones': []}
        obs = env.reset()
        done = False

        reward = 0
        ego_reward = []
        partner_reward = []
        team_reward = 0

        if render:
            env.render()
        while not done:
            action, ego_action_distr = ego.get_action(obs, False)

            # state_t = obs.squeeze(axis=1)[:-1]
            # pdb.set_trace()
            state_t = obs

            ego_observation_as_tuple = tuple(np.squeeze(env.base_env.observe('player_1'), axis=1))
            partner_observation_as_tuple = tuple(np.squeeze(env.base_env.observe('player_1'), axis=1))

            ego_action_distribution[ego_observation_as_tuple] = ego_action_distr[0]

            output = env.step(action, with_team_reward=True)
            # print("output", output)
            obs, _, newreward, done, info, all_actions, partner_action_distr, _, action_successful = output

            partner_action_distr = partner_action_distr[0]
            partner_action_distribution[partner_observation_as_tuple] = partner_action_distr

            # print("done", done)
            player_actions = number_to_action[all_actions[0][0]], number_to_action[all_actions[1][0]]

            ego_reward.append(newreward[0])
            partner_reward.append(newreward[1])

            reward += newreward[0]

            game_results[game]['observations'].append(state_t)
            game_results[game]['rewards'].append(ego_reward)
            game_results[game]['actions'].append(player_actions)
            game_results[game]['dones'].append(done)

            game_history.append((all_actions[0][0], all_actions[1][0]))

            if done:
                # team_reward = ego_reward + partner_reward
                team_reward = ego_reward[-1] + partner_reward[-1]

            # if render:
            #     env.render()
            #     sleep(1 / 60)
        # pdb.set_trace()
        ego_rewards.append(ego_reward[-1])
        partner_rewards.append(partner_reward[-1])

        team_rewards.append(team_reward)

    env.close()

    print(f"Average Reward: {sum(ego_rewards) * 1.0 / (num_episodes)}")
    print(f"Standard Deviation: {np.std(ego_rewards)}")

    prediction_accuracy = 0
    return prediction_accuracy, partner_action_distribution, ego_action_distribution, ego_rewards, partner_rewards, team_rewards

def test_trained_partner_model(game_name, partner_params, action_to_one_hot, max_cycles, num_episodes, number_to_action):
    env, altenv = generate_env(max_cycles)
    print(f"Environment: {env}; Partner env: {altenv}")

    ego = gen_fixed("PPO", f'{game_name}/saved_models/policy')

    alt = gen_fixed("PPO", f'{game_name}/saved_models/partner_{0}')
    env.add_partner_agent(alt)

    input_size = partner_params['input_size']
    hidden_size = partner_params['hidden_size']
    num_layers = partner_params['num_layers']
    num_classes = env.action_space.n  # 4

    # self.partner_model = HumanLSTM(num_classes, input_size, hidden_size, num_layers)
    ego_partner_model = HumanLSTM(num_classes, input_size, hidden_size, num_layers)
    ego_partner_model.load_state_dict(torch.load(f'{game_name}/saved_models/partner_model'))
    # ego_partner_model.double()
    ego_partner_model.to(device=DEVICE)
    ego_partner_model.eval()

    # Run game

    ego_rewards = []
    partner_rewards = []
    team_rewards = []

    game_results = {}

    action_distribution_ego = {action_idx: 0 for action_idx in range(num_classes)}
    action_distribution_partner =  {action_idx: 0 for action_idx in range(num_classes)}

    total_correct_predictions = 0
    total_predictions_made = 0

    partner_obs = None
    partner_action_distr = None
    ego_action_distr = None

    partner_action_distribution = {}
    ego_action_distribution = {}

    render = False
    state_t_minus_1 = None
    game_history = []
    for game in range(num_episodes):
        game_results[game] = {'observations': [], 'rewards': [], 'actions': [], 'dones': []}
        obs = env.reset()
        done = False

        reward = 0
        ego_reward = []
        partner_reward = []
        team_reward = 0



        if render:
            env.render()
        while not done:
            action, ego_action_distr = ego.get_action(obs, False)

            # state_t = obs.squeeze(axis=1)[:-1]
            # pdb.set_trace()
            state_t = obs



            # print("Ego action", number_to_action[action])
            # pdb.set_trace()
            ego_observation_as_tuple = tuple(np.squeeze(env.base_env.observe('player_1'), axis=1))
            partner_observation_as_tuple = tuple(np.squeeze(env.base_env.observe('player_1'), axis=1))
            # print("ego_observation_as_tuple", ego_observation_as_tuple)
            # print("partner_observation_as_tuple", partner_observation_as_tuple)

            # obs_as_tuple = tuple(obs.squeeze(axis=1))

            ego_action_distribution[ego_observation_as_tuple] = ego_action_distr[0]

            output = env.step(action, with_team_reward=True)
            # print("output", output)
            obs, _, newreward, done, info, all_actions, partner_action_distr, _ = output
            # pdb.set_trace()
            # print(f"EGO obs: {obs}, newreward: {newreward}, ego action {action}, all_actions: {all_actions}")
            # print("partner_action_distr", partner_action_distr)
            # print("ego_action_distr", ego_action_distr)


            partner_action_distr = partner_action_distr[0]
            partner_action_distribution[partner_observation_as_tuple] = partner_action_distr


            # print("done", done)
            player_actions = number_to_action[all_actions[0][0]], number_to_action[all_actions[1][0]]

            # see human model prediction



            # input_x = []
            if state_t_minus_1 is None:
                state_t_minus_1 = np.zeros(obs.shape)
                ego_action_t_minus_1 = [0] * num_classes

            # human_pred_input_at_t = np.concatenate([ego_action_t_minus_1, state_t_minus_1, state_t], axis=0)  # length = (14,)


            # ego_action_t_minus_1.extend([0, 0, 0])
            # print("ego_action_t", ego_action_t)
            # ego_action_t_minus_1 = np.reshape(ego_action_t_minus_1, (1, 10))
            # ego_action_t_minus_1 = np.broadcast_to(ego_action_t_minus_1, (1, 10, 10))
            # pdb.set_trace()
            # print("state_t_minus_1", state_t_minus_1.shape)
            if len(state_t_minus_1.shape) == 2:
                state_t_minus_1 = np.squeeze(state_t_minus_1, axis=1)
            if len(state_t.shape) == 2:
                state_t = np.squeeze(state_t, axis=1)
            candidate_human_pred_input = np.concatenate([ego_action_t_minus_1, state_t_minus_1, state_t], axis=0)  # length = (14,)
            candidate_human_pred_input = np.expand_dims(np.array([candidate_human_pred_input]), axis=0)

            tensor_input_x = Variable(torch.Tensor(candidate_human_pred_input)).to(device=DEVICE)

            # input_x = np.expand_dims(np.array([human_pred_input_at_t]), axis=2)
            # tensor_input_x = Variable(torch.Tensor(input_x)).to(device=DEVICE)


            # print("tensor_input_x", tensor_input_x.shape)
            # pdb.set_trace()

            predicted_partner_action = ego_partner_model(tensor_input_x)
            predicted_partner_action = np.argmax(predicted_partner_action.cpu().detach().numpy()[0])
            # print("predicted partner action", predicted_partner_action)
            # print("actual partner action", all_actions[1][0])


            if predicted_partner_action == all_actions[1][0]:
                total_correct_predictions += 1
            total_predictions_made += 1

            action_distribution_ego[all_actions[0][0]] += 1
            action_distribution_partner[all_actions[1][0]] += 1

            # print("player_actions", player_actions)
            # print("obs", type(obs))
            # new_obs = number_to_action[obs.item()]
            # partner_obs = action
            state_t_minus_1 = copy.deepcopy(state_t)
            ego_action_t_minus_1 = copy.deepcopy(action_to_one_hot[action])

            ego_reward.append(newreward[0])
            partner_reward.append(newreward[1])

            reward += newreward[0]

            game_results[game]['observations'].append(state_t)
            game_results[game]['rewards'].append(ego_reward)
            game_results[game]['actions'].append(player_actions)
            game_results[game]['dones'].append(done)

            game_history.append((all_actions[0][0], all_actions[1][0]))

            if done:
                # team_reward = ego_reward + partner_reward
                team_reward = sum(ego_reward) + sum(partner_reward)

            # if render:
            #     env.render()
            #     sleep(1 / 60)
            # pdb.set_trace()
        ego_rewards.append(sum(ego_reward))
        partner_rewards.append(sum(partner_reward))

        team_rewards.append(team_reward)

    env.close()

    action_distribution_ego = {action_distribution_ego[elem] / sum(action_distribution_ego.values()) for elem in
                               action_distribution_ego}
    action_distribution_partner = {action_distribution_partner[elem] / sum(action_distribution_partner.values()) for
                                   elem in
                                   action_distribution_partner}
    print(f"Average Reward: {sum(ego_rewards) * 1.0 / (num_episodes)}")
    print(f"Standard Deviation: {np.std(ego_rewards)}")

    print(f"Percent correct predictions {total_correct_predictions/total_predictions_made}, "
          f"total correct {total_correct_predictions} out of {total_predictions_made}")

    prediction_accuracy = total_correct_predictions/total_predictions_made
    return prediction_accuracy, partner_action_distribution, ego_action_distribution, ego_rewards, partner_rewards, team_rewards


def plot_over_time_distribution(game, desired_strategy, over_time_distribution, savename=''):
    # pdb.set_trace()
    # for observation in [0,1]:
    observation_list = []
    for timestep in over_time_distribution:
        for keyname in over_time_distribution[timestep]:
            # pdb.set_trace()
            if (keyname[0], keyname[1]) not in observation_list:
                observation_list.append((keyname[0], keyname[1]))

    for observation in observation_list:
        plt.figure()
        patches_list = []
        for timestep in over_time_distribution:
            timestep_color = "#" + ''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)])
            patch_t = mpatches.Patch(color=timestep_color, label=str(timestep))
            patches_list.append(patch_t)

            for keyname in over_time_distribution[timestep]:
                if keyname[0] == observation[0] and keyname[1] == observation[1]:
                    action_distr = over_time_distribution[timestep][keyname]
                    action_distr = list(action_distr)

                    plt.plot(range(len(action_distr)), action_distr, color=timestep_color)
                    # break

        plt.legend(handles=patches_list)

        plt.title(f'Action Probs Given Observation {observation}: {savename}')
        plt.xlabel("Action Index")
        plt.ylabel(f"Probability Given Observation {observation}")
        plt.savefig(f'{game}/images/observations/observation_{observation}_' + savename + '.png')
        plt.close()


def plot_over_time_distribution_as_time_series(game, over_time_distribution, number_to_action, number_to_color,
                                               savename=''):
    # pdb.set_trace()
    # for observation in [0,1]:
    observation_list = []
    for timestep in over_time_distribution:
        for keyname in over_time_distribution[timestep]:
            # print("keyname", keyname)
            # pdb.set_trace()
            current_state = keyname
            if current_state not in observation_list:
                observation_list.append(current_state)
            # if (keyname[0], keyname[1]) not in observation_list:
            #     observation_list.append((keyname[0], keyname[1]))

    for o in (range(len(observation_list))):
        observation = observation_list[o]
        plt.figure()

        action_num_to_prob_list = {a: [1 / len(number_to_action)] for a in number_to_action}

        for timestep in over_time_distribution:
            avg_over_keys = {a: [] for a in number_to_action}
            for full_keyname in over_time_distribution[timestep]:
                keyname = full_keyname
                # if keyname[0] == observation[0] and keyname[1] == observation[1]:
                if keyname == observation:
                    action_distr = over_time_distribution[timestep][keyname]

                    for idx in range(len(action_distr)):
                        avg_over_keys[idx].append(action_distr[idx])

            for a in number_to_action:
                action_num_to_prob_list[a].append(
                    (np.mean(avg_over_keys[a]) if len(avg_over_keys[a]) > 0 else action_num_to_prob_list[a][-1]))

        for a in number_to_action:
            # print("action_num_to_prob_list[a]", action_num_to_prob_list[a])
            # pdb.set_trace()
            plt.plot(range(len(action_num_to_prob_list[a])), action_num_to_prob_list[a], color=number_to_color[a],
                     label=number_to_action[a])

        plt.legend()

        plt.title(f'Obs {observation}: {savename}')
        plt.xlabel("Interaction Number")
        plt.ylabel(f"Probability Given Observation")
        plt.savefig(f'{game}/images/observations/observation_{str(observation)}_' + savename + '.png')
        plt.close()

        # with open(f'{game}/images/observations/observation_{str(observation)}_'+savename+'.txt', "w") as text_file:
        #     text_file.write(str(observation))


def transform_influence_reward(env_reward, mi_divergence, mi_guiding_divergence):
    alpha = 1
    beta = 10
    gamma = 20
    # final_reward = alpha * env_reward + beta * mi_divergence - gamma * mi_guiding_divergence
    # final_reward = alpha * env_reward - gamma * mi_guiding_divergence
    final_reward = alpha * env_reward + beta * mi_divergence
    # final_reward = beta * mi_divergence
    # final_reward = env_reward
    return final_reward


if __name__ == '__main__':
    # game = 'collab-particle-v2_env-div-rew_influence_exp8_lstm'
    game = 'harvest_vector_v0_truepartner_exp5_beta10_15iters_rew-env-inf'
    print(f"Running Experiment: {game}")

    # Create folder for experiment data
    if not os.path.exists(game):
        # Create a new directory because it does not exist
        os.makedirs(game)

        # Create a saved models and images directory
        os.makedirs(f'{game}/saved_models')
        os.makedirs(f'{game}/images')
        os.makedirs(f'{game}/images/observations')
        os.makedirs(f'{game}/training_history')
        print(f"Created folder for experiment: {game}")

    # current position: action distribution
    # desired_strategy_as_str = {}
    # desired_strategy = None
    # if desired_strategy is not None:
    #     for key_tuple in desired_strategy:
    #         desired_strategy_as_str[str(key_tuple)] = desired_strategy[key_tuple]

    game_params = {
        'device': 'cuda:0',
        'transform_influence_reward': transform_influence_reward,
        'game_name': game,
        'max_cycles': 50,
        # 'desired_strategy': [0.05, 0.05, 0.05, 0.05, 0.0, 0.8, 0.0],
        'desired_strategy': [0.25, 0.25, 0.24, 0.24, 0.01, 0.01],
        'num_iterations_per_ep': 10000,
        'num_interaction_episodes': 10,
        'num_test_games': 5,
        'action_to_one_hot': {0: [1, 0, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0, 0], 3: [0, 0, 0, 1, 0, 0],
                              4: [0, 0, 0, 0, 1, 0], 5: [0, 0, 0, 0, 0, 1]},
        'default_marginal_probability': [0, 0, 0, 0, 0, 0],
        'number_to_action': {0: 'N', 1: 'S', 2: 'E', 3: 'W', 4: 'Consume', 5: 'Shoot'},
        'number_to_color': {0: 'r', 1: 'g', 2: 'b', 3: 'm', 4: 'k', 5: 'y'},
        'partner_params': {
            'learning_rate': 0.01,
            'input_size': 60,
            'hidden_size': 128,
            'num_layers': 1,
            'num_partner_training_epochs': 1000,
            'train_percent': 0.9,
        },
        'num_eval_games': 100,
        'rewards': '''env rew'''
    }

    # with open(f'{game}/experiment_parameters.json', 'w') as fp:
    #     json.dump(game_params, fp)

    with open(f'{game}/experiment_parameters.pickle', 'wb') as handle:
        pickle.dump(game_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    over_time_partner_action_distribution, over_time_ego_action_distribution = train(game_params)

    # print("Saving Over Time Action Distributions......")
    with open(f'{game}/training_history/over_time_partner_action_distribution.pickle', 'wb') as handle:
        pickle.dump(over_time_partner_action_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{game}/training_history/over_time_ego_action_distribution.pickle', 'wb') as handle:
        pickle.dump(over_time_ego_action_distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(f'{game}/training_history/over_time_partner_action_distribution.pickle', 'rb') as handle:
    #     over_time_partner_action_distribution = pickle.load(handle)
    #
    # with open(f'{game}/training_history/over_time_ego_action_distribution.pickle', 'rb') as handle:
    #     over_time_ego_action_distribution = pickle.load(handle)

    # plot_over_time_distribution(game, desired_strategy, over_time_partner_action_distribution, savename=f'partner_distr_over_time')
    # plot_over_time_distribution(game, desired_strategy, over_time_ego_action_distribution, savename=f'ego_distr_over_time')

    number_to_action = game_params['number_to_action']
    number_to_color = game_params['number_to_color']
    plot_over_time_distribution_as_time_series(game, over_time_partner_action_distribution, number_to_action,
                                               number_to_color,
                                               savename=f'partner_distr_over_time')
    plot_over_time_distribution_as_time_series(game, over_time_ego_action_distribution, number_to_action,
                                               number_to_color, savename=f'ego_distr_over_time')

    #
    # n_games = 10
    # game = 'rps_gamma=5_guide_only_frac_guide_influence_exp1_lstm'
    game_results = test(game_params)

    print(f"Done Running Experiment: {game}")

