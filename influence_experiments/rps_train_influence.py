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

from torch.autograd import Variable

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


# class HumanLSTM_old(nn.Module):
#     def __init__(self):
#         super(HumanLSTM_old, self).__init__()
#         self.lstm1 = nn.LSTMCell(20, 51)
#         self.lstm2 = nn.LSTMCell(51, 51)
#         self.linear = nn.Linear(51, 3)
#         # self.softmax = nn.Softmax(dim=1)
#         # self.softmax = torch.nn.functional.softmax(input, dim=1)
#         self.batch_size = 1
#
#     def forward(self, input, future = 0):
#         outputs = []
#
#         h_t = torch.zeros(self.batch_size, 51, dtype=torch.double)
#         c_t = torch.zeros(self.batch_size, 51, dtype=torch.double)
#         h_t2 = torch.zeros(self.batch_size, 51, dtype=torch.double)
#         c_t2 = torch.zeros(self.batch_size, 51, dtype=torch.double)
#
#         # for input_t in input.split(1, dim=1):
#         #     print("input_t", input_t.shape)
#         for i in range(input.shape[0]-self.batch_size):
#             input_t = input[i:i+self.batch_size, :]
#             # print("input_t", input_t.shape)
#             h_t, c_t = self.lstm1(input_t, (h_t, c_t))
#             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
#             output = self.linear(h_t2)
#             output = torch.nn.functional.softmax(output, dim=1)
#             # print("output", output.shape)
#             output = output.max(dim=1)
#             outputs += [output]
#
#         # print("outputs", outputs)
#         for i in range(future):# if we should predict the future
#             h_t, c_t = self.lstm1(output, (h_t, c_t))
#             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
#             output = self.linear(h_t2)
#             outputs += [output]
#
#         outputs = torch.cat(outputs, dim=1)
#         return outputs


class HumanLSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(HumanLSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 6

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("x", x.shape)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)
        out = self.softmax(out)

        return out

# class HumanLSTM(nn.Module):
#
#     def __init__(self, num_classes, input_size, hidden_size, num_layers):
#         super(HumanLSTM, self).__init__()
#
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         # self.seq_length = seq_length
#
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True)
#
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         pdb.set_trace()
#         h_0 = Variable(torch.zeros(
#             self.num_layers, x.size(0), self.hidden_size))
#
#         c_0 = Variable(torch.zeros(
#             self.num_layers, x.size(0), self.hidden_size))
#
#         # Propagate input through LSTM
#         ula, (h_out, _) = self.lstm(x, (h_0, c_0))
#
#         h_out = h_out.view(-1, self.hidden_size)
#
#         out = self.fc(h_out)
#
#         return out

class EgoRPSAgent():
    def __init__(self, env):
        self.ego_model = INFLUENCE_PPO('MlpPolicy', env, verbose=1)
        self.past_game_data = {}

        self.eps_index = 0
        self.seq_length = 10
        self.initialize_partner_model()
        self.batch_size = 10

    def add_game_to_train(self, eps_index, game_result):
        self.past_game_data[eps_index] = game_result
        self.eps_index = eps_index

    def initialize_partner_model(self):

        learning_rate = 0.01

        input_size = 1
        hidden_size = 2
        num_layers = 1

        num_classes = 3

        self.partner_model = HumanLSTM(num_classes, input_size, hidden_size, num_layers)
        self.partner_model.train()
        # self.partner_model.double()
        self.criterion = nn.CrossEntropyLoss()
        # use LBFGS as optimizer since we can load the whole data to train
        # self.optimizer = optim.LBFGS(self.partner_model.parameters(), lr=0.8)
        self.optimizer = optim.Adam(self.partner_model.parameters(), lr=learning_rate)

    def train_partner_model_old(self):



        train_x, train_y = self.construct_dataset()
        # train_x(2037, 20)
        # train_y(2037, 1)
        tensor_train_x = torch.from_numpy(train_x).double()
        tensor_train_y = torch.from_numpy(train_y).double()

        steps = 50
        # begin to train
        for i in range(steps):
            print('STEP: ', i)

            def closure():
                self.optimizer.zero_grad()
                out = self.partner_model(tensor_train_x)
                # print("out", out.shape)
                # print("tensor_train_y", tensor_train_y.shape)
                target = []
                for i in range(train_y.shape[0] - self.batch_size):
                    target_y = train_y[i:i + self.batch_size, 0]
                    target.append(target_y)

                target = np.array(target, dtype=np.float32)
                # print("target", target.shape)
                # print("out", out.shape)
                target = np.transpose(target, axes=(1,0))
                # print("target", target.shape)
                target = torch.from_numpy(target).double()
                loss = self.criterion(out, target)
                print(f'loss: {loss.item()}')
                loss.backward()
                return loss

            self.optimizer.step(closure)

    def train_partner_model(self):
        train_x, train_y = self.construct_dataset()
        train_x, train_y = np.expand_dims(np.array(train_x), axis=2), np.array(train_y)
        # train_x, train_y = np.array(train_x), np.array(train_y)

        dataset_size = train_x.shape[0]
        train_size = int(dataset_size * 0.67)
        test_size = dataset_size - train_size

        dataX = Variable(torch.Tensor(train_x))
        dataY = Variable(torch.Tensor(train_y))

        trainX = Variable(torch.Tensor(np.array(train_x[0:train_size])))
        trainY = Variable(torch.Tensor(np.array(train_y[0:train_size])))

        testX = Variable(torch.Tensor(np.array(train_x[train_size:dataset_size])))
        testY = Variable(torch.Tensor(np.array(train_y[train_size:dataset_size])))

        print("trainX", trainX.shape)
        print("trainY", trainY.shape)

        num_epochs = 2000
        learning_rate = 0.01

        input_size = 1
        hidden_size = 2
        num_layers = 1

        num_classes = 1

        # lstm = HumanLSTM(num_classes, input_size, hidden_size, num_layers)

        # criterion = torch.nn.MSELoss()  # mean-squared error for regression
        # optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
        # # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
        #
        # # Train the model
        # for epoch in range(num_epochs):
        #     outputs = lstm(trainX)
        #     optimizer.zero_grad()
        #
        #     # obtain the loss function
        #     loss = criterion(outputs, trainY)
        #
        #     loss.backward()
        #
        #     optimizer.step()
        #     if epoch % 100 == 0:
        #         print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        num_epochs = 1000
        # Train the model
        for epoch in range(num_epochs):
            # trainX.double()
            # trainY.double()
            # pdb.set_trace()
            outputs = self.partner_model(trainX)
            self.optimizer.zero_grad()

            # obtain the loss function
            # print("outputs", outputs)
            # print("trainY", trainY)
            trainY = trainY.type(torch.LongTensor)
            loss = self.criterion(outputs, trainY)

            loss.backward()

            self.optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



    def construct_dataset(self):
        action_to_one_hot = {0: [1,0,0], 1:[0,1,0], 2:[0,0,1]}

        train_x, train_y = [], []
        game_data = self.past_game_data[self.eps_index]
        for i in range(1, len(game_data)-1):
            add_x = []
            if 3 in game_data[i] or 3 in game_data[i+1] or 3 in game_data[i-1]:
                continue
            # add_x.extend(action_to_one_hot[game_data[i][0]])
            # add_x.extend(action_to_one_hot[game_data[i][1]])

            # state at i from partner perspective = ego action at i-1
            add_x.extend(action_to_one_hot[game_data[i-1][0]])

            # ego action at i
            add_x.extend(action_to_one_hot[game_data[i][0]])

            # print("add_x", add_x)

            # add_y = action_to_one_hot[game_data[i+1][1]]
            add_y = game_data[i+1][1]

            train_x.append(add_x)
            train_y.append(add_y)

        train_x, train_y = np.array(train_x), np.array(train_y)
        print("train_x", train_x.shape)
        print("train_y", train_y.shape)
        return train_x, train_y

def generate_ego(env):
    ego_agent = EgoRPSAgent(env)
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
            # print(f"player_actions: {player_actions}: ",(all_actions[0][0], all_actions[1][0]))

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

    # action_distribution_ego = {action_distribution_ego[elem]/sum(action_distribution_ego.values()) for elem in action_distribution_ego}
    # action_distribution_partner = {action_distribution_partner[elem] / sum(action_distribution_partner.values()) for elem in
    #                            action_distribution_partner}

    # print("action_distribution_ego", action_distribution_ego)
    # print("action_distribution_partner", action_distribution_partner)
    # pdb.set_trace()

    ego_action_sum = sum(action_distribution_ego.values())
    partner_action_sum = sum(action_distribution_partner.values())
    normalized_action_distribution_ego = {0:action_distribution_ego[0]/ego_action_sum,
                                          1:action_distribution_ego[1]/ego_action_sum,
                                          2:action_distribution_ego[2]/ego_action_sum,}

    normalized_partner_distribution_ego = {0: action_distribution_partner[0] / partner_action_sum,
                                          1: action_distribution_partner[1] / partner_action_sum,
                                          2: action_distribution_partner[2] / partner_action_sum, }

    print(f"Average Reward: {sum(rewards)*1.0/(num_episodes*15.0)}")
    print(f"Standard Deviation: {np.std(rewards)}")
    return game_results, normalized_action_distribution_ego, normalized_partner_distribution_ego

# Train Agent
def train(game_name):
    env, altenv = generate_env()
    print(f"Environment: {env}; Partner env: {altenv}")

    ego_agent = generate_ego(env)
    ego_ppo = ego_agent.ego_model
    # print(f'Ego: {ego}')
    partners = generate_partners(altenv, env, 1)

    # ego.learn(total_timesteps=100000)
    new_logger = configure('loggers/', ["stdout", "csv", "tensorboard"])
    ego_ppo.set_logger(new_logger)

    total_timesteps = 200000
    num_prediction_iters = 2000
    num_prediction_epochs = 5

    partner_prediction_accuracies = []

    for ep in range(num_prediction_epochs):
        ego_ppo.set_partner_model(ego_agent.partner_model)

        ego_ppo.set_episode_instance(ep)
        ego_ppo.learn(total_timesteps=num_prediction_iters)
        ego_ppo.save(f'saved_models/{game_name}_policy')

        # print('all_game_data_by_ep', ego_ppo.all_game_data_by_ep)
        episode_data = ego_ppo.all_game_data_by_ep[ep]
        ego_agent.add_game_to_train(ep, episode_data)
        ego_agent.train_partner_model()

        torch.save(ego_agent.partner_model.state_dict(), f'saved_models/{game_name}_partner_model')

        for i in range(len(partners)):
            partners[i].model.save(f"saved_models/{game_name}_partner_{i}")

        partner_prediction_accuracy = test_trained_partner_model(game_name, num_episodes=10)
        print(f'episode {ep}: partner_prediction_accuracy = {partner_prediction_accuracy}')
        partner_prediction_accuracies.append(partner_prediction_accuracy)


        # for rollout_data in ego_ppo.rollout_buffer.get(ego_ppo.batch_size):
        #     actions = rollout_data.actions
        #     print("rollout_data", rollout_data.shape)
        #     print("actions", actions.shape)
        #     break

    plt.plot(range(num_prediction_epochs), partner_prediction_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Partner Prediction Accuracy")
    plt.title(f'{game_name}: Partner Prediction Accuracy')
    plt.savefig(f'images/{game_name}_partner_pred_acc.png')
    plt.close()


def plot_action_distribution(game_name, action_distribution_ego, action_distribution_partner):
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    ego_dist = list(action_distribution_ego.values())
    part_dist = list(action_distribution_partner.values())

    # Set position of bar on X axis
    br1 = np.arange(3)
    br2 = [x + barWidth for x in br1]
    # br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, ego_dist, color='r', width=barWidth,
            edgecolor='grey', label='Ego')
    plt.bar(br2, part_dist, color='g', width=barWidth,
            edgecolor='grey', label='Partner')

    # Adding Xticks
    plt.xlabel('RPS Action', fontweight='bold', fontsize=15)
    plt.ylabel('Percent of Actions in 150 games', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(3)],
               ['Rock', 'Paper', 'Scissors'])

    plt.title(f"Trial {game_name}: Action Distributions")
    plt.legend()
    plt.savefig(f'images/{game_name}_action_distrib.png')
    plt.close()

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

    plot_action_distribution(game_name, action_distribution_ego, action_distribution_partner)

    return game_results, action_distribution_ego, action_distribution_partner


def test_trained_partner_model(game_name, num_episodes=10):
    env, altenv = generate_env()
    print(f"Environment: {env}; Partner env: {altenv}")

    ego = gen_fixed("PPO", f'saved_models/{game_name}_policy')

    alt = gen_fixed("PPO", f'saved_models/{game_name}_partner_{0}')
    env.add_partner_agent(alt)

    learning_rate = 0.01

    input_size = 1
    hidden_size = 2
    num_layers = 1

    num_classes = 3

    # self.partner_model = HumanLSTM(num_classes, input_size, hidden_size, num_layers)
    ego_partner_model = HumanLSTM(num_classes, input_size, hidden_size, num_layers)
    ego_partner_model.load_state_dict(torch.load(f'saved_models/{game_name}_partner_model'))
    # ego_partner_model.double()
    ego_partner_model.eval()

    # Run game
    number_to_action = {0: 'Rock', 1: 'Paper', 2: 'Scissors', 3: 'None'}

    rewards = []
    game_results = {}

    action_distribution_ego = {0: 0, 1: 0, 2: 0, 3: 0}
    action_distribution_partner = {0: 0, 1: 0, 2: 0, 3: 0}

    action_to_one_hot = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}

    total_correct_predictions = 0
    total_predictions_made = 0


    render = False
    game_history = []
    for game in range(num_episodes):
        game_results[game] = {'observations': [], 'rewards': [], 'actions': [], 'dones': []}
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

            # print(f"obs: {obs}, newreward: {newreward}, all_actions: {all_actions}")


            # print("done", done)
            player_actions = number_to_action[all_actions[0][0]], number_to_action[all_actions[1][0]]

            # see human model prediction
            input_x = []
            if len(game_history) < 2:
                input_x = [0.0] * 6
            else:
                # input_x.extend(action_to_one_hot[game_history[-1][0]])
                # input_x.extend(action_to_one_hot[game_history[-1][1]])
                input_x.extend(action_to_one_hot[game_history[-2][0]])
                input_x.extend(action_to_one_hot[game_history[-1][0]])

            input_x = np.expand_dims(np.array([input_x]), axis=2)
            tensor_input_x = Variable(torch.Tensor(input_x))


            # print("tensor_input_x", tensor_input_x.shape)

            predicted_partner_action = ego_partner_model(tensor_input_x)
            predicted_partner_action = np.argmax(predicted_partner_action.detach().numpy()[0])
            # print("predicted partner action", predicted_partner_action)
            # print("actual partner action", all_actions[1][0])

            if predicted_partner_action == all_actions[1][0]:
                total_correct_predictions += 1
            total_predictions_made += 1

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

            game_history.append((all_actions[0][0], all_actions[1][0]))

            if render:
                env.render()
                sleep(1 / 60)

        rewards.append(reward)

    env.close()

    action_distribution_ego = {action_distribution_ego[elem] / sum(action_distribution_ego.values()) for elem in
                               action_distribution_ego}
    action_distribution_partner = {action_distribution_partner[elem] / sum(action_distribution_partner.values()) for
                                   elem in
                                   action_distribution_partner}
    print(f"Average Reward: {sum(rewards) * 1.0 / (num_episodes * 15.0)}")
    print(f"Standard Deviation: {np.std(rewards)}")

    print(f"Percent correct predictions {total_correct_predictions/total_predictions_made}, "
          f"total correct {total_correct_predictions} out of {total_predictions_made}")

    prediction_accuracy = total_correct_predictions/total_predictions_made
    return prediction_accuracy


if __name__ == '__main__':
    game = 'rps_gamma=5_guide_only_frac_guide_influence_exp1_lstm'
    print(f"Running Experiment: {game}")

    # test_trained_partner_model(game, num_episodes=10)

    train(game)
    #
    # n_games = 10
    game_results = test(game, n_games=100)









