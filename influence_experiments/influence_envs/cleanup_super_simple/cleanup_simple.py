import numpy as np
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import copy
# from .board import Board

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
        "name": "harvest_v2",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(self, max_cycles=400):
        super().__init__()

        self.max_cycles = max_cycles
        # Set grid
        self.moves = ['RED', 'GREEN', 'BLUE', 'YELLOW']

        self.color_to_index = {'RED': 0, 'GREEN':1, 'BLUE':2, 'YELLOW':3}
        self.index_to_color = {0:'RED', 1:'GREEN', 2:'BLUE', 3:'YELLOW'}

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]

        self.num_actions = 4
        # self.num_agents = 2

        self.action_spaces = {i: spaces.Discrete(self.num_actions) for i in self.agents}

        self.observation_spaces = {i: spaces.Box(low=0, high=1, shape=(16, 1), dtype=np.int8) for i in self.agents}

        # print("self.observation_spaces", self.observation_spaces)

        self.rewards = {i: 0 for i in self.agents}

        self.dones = {i: False for i in self.agents}




        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._none = self.num_actions
        self.selected_actions = {agent: self._none for agent in self.agents}
        self.previous_selected_actions = {agent: self._none for agent in self.agents}
        self.num_moves = 0
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        self.infos = {i: {"legal_moves": list(range(0, self.num_actions)),
                          "partner_action": self._none} for i in self.agents}

        self.reset()
        self.observations = {}
        for i in self.agents:
            self.observations[i] = self.observe(i)



    def observe(self, agent):
        observation_list = []
        observation_list.extend(self.remaining_colors)

        partner_agent = self.agents[1 - self.agent_name_mapping[agent]]
        observation_list.extend(self.agent_past_selections[partner_agent])
        observation_list.extend(self.previous_agent_state_action[partner_agent]['state'])
        observation_list.extend(self.previous_agent_state_action[partner_agent]['action'])



        # observation = np.stack([grid_with_position, grid_with_prev_position, grid_with_lava, grid_with_destination], axis=0).astype(np.uint8)
        # obs = [self.current_position[0], self.current_position[1],
        #        self.prev_position[0], self.prev_position[1]]
        # for pos in self.lava_positions:
        #     obs.append(pos[0])
        #     obs.append(pos[1])
        # for pos in self.destinations:
        #     obs.append(pos[0])
        #     obs.append(pos[1])
        #
        # obs.append(self.previous_selected_actions[self.agents[1 - self.agent_name_mapping[agent]]])
        observation = np.array(observation_list).astype(np.int8)
        observation = np.expand_dims(observation, axis=1)

        return observation

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]



    def step(self, action, display=False):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection

        self.selected_actions[agent] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():

            # Reset Previous Action Selections
            for ag in self.agents:
                self.previous_agent_state_action[ag]['state'] = copy.deepcopy(self.remaining_colors)
                one_hot_action = [0,0,0,0]
                one_hot_action[self.selected_actions[ag]] = 1
                self.previous_agent_state_action[ag]['action'] = one_hot_action

            rewards = [-1, -1]
            if sum(self.remaining_colors) == 0:
                rewards = [0, 0]

            # If can move
            # rew = self.board.step_single_action(self.agents.index(agent), action)
            # rewards[self.agents.index(agent)] = rew
            for acting_agent in reversed(self.agents):
                selected_action = self.selected_actions[acting_agent]

                # If can remove object
                if self.remaining_colors[selected_action] > 0:
                    self.remaining_colors[selected_action] -= 1

                    # Update selection counter
                    self.agent_past_selections[acting_agent][self.selected_actions[acting_agent]] += 1

                    rew = self.agent_reward_weights[acting_agent][selected_action]
                    rewards[self.agents.index(acting_agent)] = rew
                    # rewards[1-self.agents.index(acting_agent)] = 0

                # else:
                #     rew = 0
                #     rewards[self.agents.index(acting_agent)] = rew



            # Increment timestep
            self.current_timestep += 1


            # If last timestep, add team reward
            if self.current_timestep > self.max_timesteps or sum(self.remaining_colors) == 0:
                # No final reward for now
                self.dones = {
                    a: True for a in self.agents
                }


            self.rewards[self.agents[0]], self.rewards[self.agents[1]] = tuple(rewards)

            self.num_moves += 1
            # observe the current state
            for i in self.agents:
                self.observations[i] = self.observe(i)


            for a in self.agents:
                self.infos[a]['partner_action'] = self.selected_actions


        else:
            self.selected_actions[self.agents[1 - self.agent_name_mapping[agent]]] = self._none
            self._clear_rewards()

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def reset(self, seed=None, return_info=False, options=None):
        # reset environment
        self.remaining_colors = [3, 3, 3, 3]

        self.previous_agent_state_action = {agent: {'state': [3,3,3,3],
                                                      'action': [0,0,0,0]}
                                              for agent in self.agents}

        self.agent_past_selections = {agent: [0,0,0,0] for agent in self.agents}

        self.selected_actions = {agent: self._none for agent in self.agents}
        self.num_moves = 0

        self.current_timestep = 0
        self.max_timesteps = 60
        self.agent_reward_weights = {agent: [10,10,10,10] for agent in self.agents}
        # self.agent_reward_weights = {"player_1": [0, 0, 10, 10], "player_2": [10, 10, 10, 10]}

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

    def render(self, mode="human"):

        print("REMAINING COLORS:\n", self.remaining_colors)

    def close(self):
        pass