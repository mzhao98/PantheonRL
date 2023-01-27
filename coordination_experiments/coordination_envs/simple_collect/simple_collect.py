import pdb

import numpy as np
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import copy
import matplotlib
import matplotlib.pyplot as plt
import copy
import imageio
from operator import itemgetter
import os

from .board import Board

#
# def env():
#     env = raw_env()
#     env = wrappers.CaptureStdoutWrapper(env)
#     env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
#     env = wrappers.AssertOutOfBoundsWrapper(env)
#     env = wrappers.OrderEnforcingWrapper(env)
#     return env

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "simple_collect_v0",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(self, max_cycles=400):
        super().__init__()

        self.max_cycles = max_cycles
        # Set grid
        with_reward_shaping = False
        self.board = Board(with_reward_shaping)


        self.grid_shape = self.board.grid_shape



        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]
        self.num_actions = 5
        # self.num_agents = 2

        self.action_spaces = {i: spaces.Discrete(self.num_actions) for i in self.agents}
        # self.observation_spaces = {
        #     i: spaces.Dict(
        #         {
        #             "observation": spaces.Box(
        #                 low=0, high=1, shape=(3, self.grid_shape[0], self.grid_shape[1]), dtype=np.int8
        #             ),
        #             "action_mask": spaces.Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8),
        #         }
        #     )
        #     for i in self.agents
        # }
        # self.observation_spaces = {i: spaces.Box(low=0, high=1, shape=(25, 1), dtype=np.int8) for i in self.agents}
        # self.observation_spaces = {i: spaces.Box(low=0, high=255, shape=(self.grid_shape[0], self.grid_shape[1], 3), dtype=np.uint8) for i in self.agents}
        # self.observation_spaces = {
        #     i: spaces.Box(low=0, high=1, shape=(3, self.grid_shape[0], self.grid_shape[1]), dtype=np.int8) for i in
        #     self.agents}
        self.observation_spaces = {i: spaces.Box(low=0, high=1, shape=(29, 1), dtype=np.int8) for i in self.agents}


        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}


        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._none = self.num_actions
        self.selected_actions = {agent: self._none for agent in self.agents}
        self.selected_actions_successful = {agent: 0 for agent in self.agents}
        self.previous_selected_actions = {agent: self._none for agent in self.agents}
        self.num_moves = 0
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        self.infos = {i: {"legal_moves": list(range(0, self.num_actions)),
                          "partner_action": self._none,
                          'successful_action': 0} for i in self.agents}

        self.observations = {}
        for i in self.agents:
            self.observations[i] = self.observe(i)

        self.display_counter = 0

    def observe(self, agent):
        # print(f"agent = {agent}, self.agents.index(agent) = {self.agents.index(agent)}")
        observation = self.board.observation_as_vector(self.agents.index(agent))

        return observation

    def get_partner_action(self, agent):
        partner_action = self.previous_selected_actions[self.agents[1 - self.agents.index(agent)]]
        return partner_action

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action, display=False, display_savename='test'):
        if self.dones[self.agent_selection]:
            if display:
                self.save_to_gif(display_savename)

            return self._was_done_step(action)
        agent = self.agent_selection

        self.selected_actions[agent] = action



        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # print(f"game pos: {self.current_position}, selected actions: {self.selected_actions}, prev actions: {self.previous_selected_actions}")

            self.previous_selected_actions = copy.deepcopy(self.selected_actions)

            rewards = [0, 0]
            if display:
                self.render(f'test_{self.display_counter}')

            # If can move
            # print("joint action = ", self.selected_actions)
            collective_rew = self.board.step_joint_action([self.selected_actions[acting_agent] for acting_agent in self.agents])
            # print("self.board.player_positions", self.board.player_positions)
            for acting_agent in self.agents:
                rewards[self.agents.index(acting_agent)] += collective_rew

            # Increment timestep
            self.board.current_timestep += 1


            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = tuple(rewards)

            self.num_moves += 1
            # observe the current state
            for i in self.agents:
                self.observations[i] = self.observe(i)

            self.dones = {
                a: self.board.is_done() for a in self.agents
            }
            for a in self.agents:
                self.infos[a]['partner_action'] = self.selected_actions
                self.infos[a]['successful_action'] = self.selected_actions_successful[a]

            # print("self.previous_selected_actions", self.previous_selected_actions)
            self.board.set_previous_actions(self.previous_selected_actions)


        else:
            self.selected_actions[self.agents[1 - self.agent_name_mapping[agent]]] = self._none
            self._clear_rewards()
            self.selected_actions_successful = {agent: 0 for agent in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def reset(self, seed=None, return_info=False, options=None):
        # reset environment
        self.board.reset()

        self.selected_actions = {agent: self._none for agent in self.agents}
        self.selected_actions_successful = {agent: 0 for agent in self.agents}
        self.previous_selected_actions = {agent: self._none for agent in self.agents}
        self.num_moves = 0

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {"legal_moves": list(range(0, self.num_actions)),
                          "partner_action": self._none,
                          'successful_action': 0} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

        self.observations = {}
        for i in self.agents:
            self.observations[i] = self.observe(i)

    def save_to_gif(self, savename):
        images = []
        for filename in [f'imgs_for_gifs/img_{t}.png' for t in range(len(data))]:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{savename}.gif', images)

    def render(self, savename, mode='human'):
        colors = ['white', 'black', 'red', 'blue', 'yellow', 'green', 'magenta', 'orange', 'brown']
        cmap = matplotlib.colors.ListedColormap(colors, name='colors', N=None)

        current_map = copy.deepcopy(self.board.grid)

        if self.board.doors_open is True:
            for door_loc in self.board.door_locations:
                current_map[door_loc] = 0

        for gem_loc in self.board.gem_locations:
            if gem_loc not in self.board.gems_remaining:
                current_map[gem_loc] = 0

        p1_row, p1_col = self.board.player_positions[0]
        p2_row, p2_col = self.board.player_positions[1]
        current_map[self.board.player_positions[0]] = 2
        current_map[self.board.player_positions[1]] = 3



        plt.imshow(current_map, cmap=cmap)
        plt.savefig(f'{savename}.png',
                    transparent=False,
                    facecolor='white'
                    )
        plt.close()


        if 'done' in savename:
            max_num_iters = int(savename.split('_')[-1])
            partial_file = savename.split('done')[0]
            images = []
            for filename in [f'{partial_file}{t}.png' for t in range(max_num_iters)]:
                images.append(imageio.imread(filename))
            imageio.mimsave(f'{savename}.gif', images)

            for filename in [f'{partial_file}{t}.png' for t in range(max_num_iters)]:
                os.remove(filename)
            # self.save_to_gif(savename)

        return

    def close(self):
        pass